#!/usr/bin/env python3
"""
make_veh_cls_from_video.py
---------------------------------
Turn a traffic video into a **vehicle classification dataset** folder tree:

veh_cls/
  train/
    motorcycle/
    car/
    van/
    lorry/
    bus/
    pickup/
  val/
    ...
  test/
    ...

It uses your **custom YOLO detector** (best.pt trained on the 6 classes)
to detect vehicles, then crops each detection and saves it to the class folder.
Great for training a Stage-2 classifier (yolo11n-cls / yolo11s-cls).

REQUIREMENTS
- pip install ultralytics opencv-python numpy

USAGE (typical)
python make_veh_cls_from_video.py \
  --video traffic.mp4 \
  --weights runs/detect/veh_v1/weights/best.pt \
  --out veh_cls \
  --every-sec 0.5 \
  --imgsz 1280 \
  --conf 0.25 --iou 0.6 \
  --split "80,10,10" \
  --min-box-area 900

Notes
- This expects your detector was trained on EXACT classes:
  ["motorcycle", "car", "van", "lorry", "bus", "pickup"]
- If you pass a COCO model by mistake, unknown class names will be skipped.
- Use --roi to restrict to roadway region (normalized x1,y1,x2,y2 in [0,1]).
- To avoid floods of near-duplicates, there's a temporal de-duplication (IoU filter).

"""

import argparse, os, csv, math, random, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["motorcycle", "car", "van", "lorry", "bus", "pickup"]
CLASS_SET = set(CLASSES)

def parse_roi(roi_str, W, H):
    """Parse normalized ROI 'x1,y1,x2,y2' into pixel coords."""
    if not roi_str:
        return (0, 0, W, H)
    f = [float(x.strip()) for x in roi_str.split(",")]
    if len(f) != 4:
        raise ValueError("--roi must have 4 comma-separated floats in [0,1]")
    fx1, fy1, fx2, fy2 = [min(1.0, max(0.0, v)) for v in f]
    x1, y1 = int(fx1*W), int(fy1*H)
    x2, y2 = int(fx2*W), int(fy2*H)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, W, H)
    return (x1, y1, x2, y2)

def ensure_tree(out_dir):
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            (Path(out_dir)/split/c).mkdir(parents=True, exist_ok=True)

def split_sampler(split_str):
    """Return a function that picks 'train'/'val'/'test' with given ratios."""
    parts = [int(x) for x in split_str.split(",")]
    if len(parts) != 3 or sum(parts) == 0:
        parts = [80, 10, 10]
    total = sum(parts)
    probs = [p/total for p in parts]
    bins = np.cumsum(probs)
    names = ["train", "val", "test"]
    def sample():
        r = random.random()
        for i, b in enumerate(bins):
            if r <= b:
                return names[i]
        return names[-1]
    return sample, {"train":parts[0], "val":parts[1], "test":parts[2]}

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter + 1e-9
    return inter / union

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Input video file")
    ap.add_argument("--weights", type=str, required=True, help="YOLO detector weights (custom 6-class)")
    ap.add_argument("--out", type=str, default="veh_cls", help="Output dataset root")
    ap.add_argument("--every-sec", type=float, default=0.5, help="Take a frame every N seconds")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    ap.add_argument("--device", type=str, default=None, help="e.g. '0' or 'cpu'")
    ap.add_argument("--min-box-area", type=int, default=900, help="Ignore tiny detections (pixels^2)")
    ap.add_argument("--max-per-frame", type=int, default=50, help="Guardrail to limit crops per frame")
    ap.add_argument("--split", type=str, default="80,10,10", help="Split ratios train,val,test")
    ap.add_argument("--roi", type=str, default=None, help="Normalized ROI 'x1,y1,x2,y2' in [0,1]")
    ap.add_argument("--dedup-iou", type=float, default=0.9, help="Skip saving if IoU with a recent saved box > this")
    ap.add_argument("--dedup-window", type=int, default=30, help="Keep last N saved boxes per class for dedup (per video)")
    ap.add_argument("--limit-per-class", type=int, default=0, help="Stop after saving N images per class (0=unlimited)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(math.ceil(fps * args.every_sec)))
    print(f"[INFO] fps={fps:.2f}, total_frames={total_frames}, step={step}")

    # Grab first frame to compute ROI in pixels
    ret, frame0 = cap.read()
    if not ret:
        raise SystemExit("No frames in the video.")
    H, W = frame0.shape[:2]
    roi_px = parse_roi(args.roi, W, H)
    x1r, y1r, x2r, y2r = roi_px
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Prepare dataset folders + meta CSV
    ensure_tree(args.out)
    meta_dir = Path(args.out) / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    csv_path = meta_dir / f"{Path(args.video).stem}_crops.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["split","class","filepath","video","frame_idx","time_sec","x1","y1","x2","y2","conf"])

    # Load detector
    model = YOLO(args.weights)
    names = model.names
    print(f"[INFO] Loaded detector with classes: {names}")

    # Verify class names (warn if mismatch)
    det_classes = set([names[k] for k in sorted(names.keys())])
    missing = CLASS_SET - det_classes
    if missing:
        print(f"[WARN] Your detector is missing classes {sorted(list(missing))}. "
              f"Detections of unknown classes will be skipped.")

    # Per-class counters and dedup buffers
    counts = {c: 0 for c in CLASSES}
    buffers = {c: [] for c in CLASSES}  # recent boxes for dedup (list of (box))

    frame_idx = -1
    saved_total = 0

    while True:
        ret = cap.grab()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

        # Predict on full frame; filter later via ROI
        res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=args.device, verbose=False)[0]

        saved_this_frame = 0
        for b in res.boxes:
            cls_id = int(b.cls.item())
            cls_name = names.get(cls_id, None)
            if cls_name not in CLASS_SET:
                continue

            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            # ROI filter (center must be inside ROI)
            cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
            if not (x1r <= cx <= x2r and y1r <= cy <= y2r):
                continue

            # Area filter
            box_area = max(0, x2-x1) * max(0, y2-y1)
            if box_area < args.min_box_area:
                continue

            # Per-class limit
            if args.limit_per_class > 0 and counts[cls_name] >= args.limit_per_class:
                continue

            # Dedup against recent saved boxes for this class
            recent = buffers[cls_name]
            if any(iou((x1,y1,x2,y2), rb) >= args.dedup_iou for rb in recent):
                continue

            # Decide split
            split = sampler()

            # Save crop
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ts = frame_idx / fps
            save_dir = Path(args.out) / split / cls_name
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{Path(args.video).stem}_f{frame_idx:06d}_{cls_name}_{int(ts*1000):08d}.jpg"
            out_path = save_dir / fname
            cv2.imwrite(str(out_path), crop)

            # Log meta
            conf = float(b.conf.item())
            csv_writer.writerow([split, cls_name, str(out_path), str(args.video), frame_idx, f"{ts:.3f}",
                                 x1,y1,x2,y2, f"{conf:.4f}"])

            counts[cls_name] += 1
            saved_total += 1
            saved_this_frame += 1
            recent.append((x1,y1,x2,y2))
            if len(recent) > args.dedup_window:
                recent.pop(0)

            if saved_this_frame >= args.max_per_frame:
                break

    cap.release()
    csv_file.close()

    print("\n=== SUMMARY ===")
    for c in CLASSES:
        print(f"{c:10s}: {counts[c]}")
    print(f"saved_total: {saved_total}")
    print(f"meta_csv: {csv_path}")
    print("Done.")

if __name__ == "__main__":
    # build sampler once
    # we'll parse args twice: once to get split string for sampler, then again in main
    import sys, argparse, numpy as np, random
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--split", type=str, default="80,10,10")
    ap.add_argument("--seed", type=int, default=42)
    known, _ = ap.parse_known_args()
    random.seed(known.seed); np.random.seed(known.seed)
    sampler, ratios = split_sampler(known.split)
    # run main with full args
    main()
