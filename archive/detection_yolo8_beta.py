#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic YOLO detect + optional YOLO11 classification refine (single file).

- Detector: your custom 6-class YOLO (best.pt) with names:
    {0:'motorcycle', 1:'car', 2:'van', 3:'lorry', 4:'bus', 5:'pickup'}
- Classifier: YOLO11 classification checkpoint (prefer your fine-tuned 6-class model).
  Runs on detection crops to refine labels (default: only when detector says 'lorry').

Controls/UI:
- File dialog to select a video OR type an RTSP URL.
- Buttons: Poly / Start / Line / Exit (click with mouse)
- Keys:  d=down, u=up, b=both   (count direction)
         l=line edit, a=auto line from ROI, r=reset polygon
         c=capture snapshot, t=toggle stream, ESC=quit
"""

import cv2
import torch
import numpy as np
import time
import os, sys, re, json, shutil, subprocess, traceback
from urllib.parse import urlparse
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, simpledialog
from collections import defaultdict, OrderedDict
from ultralytics import YOLO

# =========================
# Paths (SET THESE)
# =========================
DET_WEIGHTS = "best.pt"     # your 6-class detector
# If you haven't fine-tuned a classifier yet, this can point to yolo11n-cls.pt
# but it will output ImageNet classes, not your 6 classes (train one ASAP).
CLS_WEIGHTS = "runs/classify/train2/weights/best.pt"  # your trained 6-class classifier (recommended)

# Classifier usage
USE_CLASSIFIER = True
CLS_IMGSZ = 224
# Refine only certain detector labels (e.g., split 'lorry' better); set to None to refine ALL.
REFINE_ONLY = {"lorry"}

# Debug prints (first inference)
DEBUG = True
_printed_chk = False

# =========================
# Global perf switches
# =========================
cv2.setUseOptimized(True)
cv2.setNumThreads(0)
try:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

# =========================
# Streaming helper
# =========================
class FFmpegStreamer:
    """Push BGR frames to FFmpeg which publishes to RTSP/RTMP."""
    def __init__(self, url, fps, width, height, use_nvenc=False):
        self.url = url
        self.proc = None
        self.width = width
        self.height = height
        self.fps = int(round(fps)) if fps else 25
        vcodec = 'h264_nvenc' if use_nvenc else 'libx264'
        out_format = 'rtsp' if url.startswith('rtsp://') else 'flv'  # RTMP uses flv muxer
        args = [
            'ffmpeg','-re','-f','rawvideo','-pix_fmt','bgr24',
            '-s:v', f'{self.width}x{self.height}','-r', str(self.fps),
            '-i','-','-an','-c:v', vcodec,
        ]
        if use_nvenc:
            args += ['-preset','p3','-tune','ull','-rc','vbr','-b:v','2M','-g', str(self.fps)]
        else:
            args += ['-preset','veryfast','-tune','zerolatency','-b:v','2M','-g', str(self.fps)]
        args += ['-pix_fmt','yuv420p']
        if out_format == 'rtsp':
            args += ['-f','rtsp','-rtsp_transport','tcp', self.url]
        else:
            args += ['-f', out_format, self.url]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE)

    def write(self, bgr_frame: np.ndarray):
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(bgr_frame.tobytes())
        except (BrokenPipeError, IOError):
            self.close()

    def close(self):
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except:
                pass
            try:
                self.proc.terminate()
            except:
                pass
            self.proc = None

# =========================
# Centroid Tracker
# =========================
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_distance = max_distance
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        if objectID in self.objects: del self.objects[objectID]
        if objectID in self.disappeared: del self.disappeared[objectID]
    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > 10:
                    self.deregister(objectID)
            return self.objects
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row); usedCols.add(col)
            unusedCols = set(range(len(input_centroids))) - usedCols
            for col in unusedCols:
                self.register(input_centroids[col])
        return self.objects

def ffmpeg_has_nvenc():
    try:
        out = subprocess.check_output(['ffmpeg','-hide_banner','-encoders'], text=True, stderr=subprocess.STDOUT, timeout=4)
        return ('h264_nvenc' in out) or ('hevc_nvenc' in out)
    except Exception:
        return False

def gpu_availability_report():
    info = {"torch_cuda_available": None,"torch_cuda_version": None,"gpu_count": 0,"gpu_names": [],
            "gpu_caps": [],"gpu_mem_gb": [],"nvidia_smi": None}
    try:
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if info["torch_cuda_available"]:
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                info["gpu_names"].append(props.name)
                info["gpu_caps"].append(f"{props.major}.{props.minor}")
                info["gpu_mem_gb"].append(round(props.total_memory / (1024**3), 2))
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        out = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
                                      text=True, stderr=subprocess.STDOUT, timeout=3)
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        info["nvidia_smi"] = lines
    except Exception as e:
        info["nvidia_smi_error"] = str(e)
    info["ffmpeg_nvenc"] = ffmpeg_has_nvenc()
    print("[GPU] Torch CUDA available:", info.get("torch_cuda_available"))
    print("[GPU] Torch CUDA build   :", info.get("torch_cuda_version"))
    if info.get("gpu_count", 0) > 0:
        for i, (n, cap, mem) in enumerate(zip(info["gpu_names"], info["gpu_caps"], info["gpu_mem_gb"])):
            print(f"[GPU] #{i}: {n} (CC {cap}), {mem} GB")
    if info.get("nvidia_smi"):
        print("[GPU] nvidia-smi:", " | ".join(info["nvidia_smi"]))
    elif "nvidia_smi_error" in info:
        print("[GPU] nvidia-smi not available:", info["nvidia_smi_error"])
    print("[GPU] FFmpeg NVENC:", "available" if info["ffmpeg_nvenc"] else "not found")
    return info

def select_device():
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _ = torch.cuda.get_device_name(0)
            return "cuda:0"
    except Exception as e:
        print(f"[CUDA] Falling back to CPU: {e}", file=sys.stderr)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    return "cpu"

DEVICE = select_device()
print(f"[DEVICE] Using {DEVICE}")

# =========================
# YOLO models
# =========================
# --- DETECTOR ---
det = YOLO(DET_WEIGHTS)
print(f"[CHK] det.task = {det.task}")
if det.task != "detect":
    raise SystemExit(f"DET_WEIGHTS is a '{det.task}' model (need 'detect'). "
                     f"Point to your 6-class DETECTOR weights, not a classifier: {DET_WEIGHTS}")
print("[CHK] det.names:", det.names)

# --- CLASSIFIER (optional) ---
clf = None
if USE_CLASSIFIER:
    if os.path.exists(CLS_WEIGHTS):
        clf = YOLO(CLS_WEIGHTS)
        print(f"[CHK] classifier loaded: {CLS_WEIGHTS}")
    else:
        print(f"[WARN] Classifier weights not found: {CLS_WEIGHTS} — disabling classifier.")
        USE_CLASSIFIER = False

# =========================
# UI / ROI / counting state
# =========================
running = True
start_processing = False

polygon_points = []
polygon_defined = False
poly_editing = False
polygon_mask = None
current_mouse = (0, 0)

counting_line_y = 250
line_mode = 'AUTO'
line_edit_mode = False
COUNT_DIR = 'down'   # 'down' (default), 'up', 'both'

vehicle_count_by_class = defaultdict(int)
vehicle_count_by_group = defaultdict(int)

output_dir = "vehicle_captures"
counted_ids = set()
tracking_trails = defaultdict(list)
tracker = CentroidTracker()

last_crop = None
last_label = ""
do_capture = False

track_info = {}   # objectID -> {"last_centroid": (x,y), "counted": False, "last_y_bottom": None}

# Time setup defaults
time_obj = datetime.strptime("08:00:00", "%H:%M:%S").time()
time_end_obj = datetime.strptime("09:00:00", "%H:%M:%S").time()
start_date = datetime.today().date()
p1 = datetime.combine(start_date, time_obj).strftime("%d/%m/%Y")
start_time = datetime.combine(start_date, time_obj)
capture_time = datetime.combine(start_date, time_obj)
p2 = start_time.strftime("%H:%M:%S")
end_time = datetime.combine(start_date, time_end_obj)
p3 = end_time.strftime("%H:%M:%S")
p4 = start_time.strftime("%H:%M:%S")

# Min size thresholds (per label)
size_thresholds = {
    'motorcycle': (5, 10),
    'car': (30, 50),
    'van': (30, 50),
    'bus': (60, 100),
    'pickup': (40, 90),
    'lorry': (50, 100),
}

# Colors for drawing per class (BGR)
COLOR_MAP = {
    "motorcycle": (200,200,200),
    "car":        (0,255,255),
    "van":        (255,0,0),
    "lorry":      (128,0,128),
    "bus":        (0,0,255),
    "pickup":     (0,255,0),
}

# =========================
# Helpers
# =========================
def make_safe_dir_name(source: str) -> str:
    if "://" in source:
        u = urlparse(source)
        base = (u.hostname or "live") + (u.path or "")
        base = base.replace("/", "_")
    else:
        base = os.path.splitext(os.path.basename(source))[0]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    return safe or "capture"

def polygon_mid_y(pts):
    ys = [p[1] for p in pts]
    return (min(ys) + max(ys)) // 2 if ys else 250

def rebuild_polygon_mask(frame_shape):
    global polygon_mask, counting_line_y
    polygon_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if len(polygon_points) >= 3:
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(polygon_mask, [pts], 255)
        if line_mode == 'AUTO':
            counting_line_y = polygon_mid_y(polygon_points)

def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def point_in_polygon(pt):
    if not polygon_defined or len(polygon_points) < 3:
        return True
    pts = np.array(polygon_points, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0

def draw_btn(img, rect, label, color):
    (x1, y1, x2, y2) = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.putText(img, label, (x1 + 8, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def rect_with_opacity(img, pt1, pt2, color=(0, 0, 0), alpha=0.5, border_thickness=0, border_color=(255,255,255)):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    if border_thickness > 0:
        cv2.rectangle(img, pt1, pt2, border_color, thickness=border_thickness)
    return img

def draw_polygon_and_line(img):
    if len(polygon_points) > 0:
        for pt in polygon_points:
            cv2.circle(img, pt, 3, (0, 255, 255), -1)
        for i in range(1, len(polygon_points)):
            cv2.line(img, polygon_points[i-1], polygon_points[i], (0, 255, 255), 2)
        if poly_editing:
            cv2.line(img, polygon_points[-1], current_mouse, (0, 255, 255), 1)
    if polygon_defined and polygon_mask is not None:
        overlay = img.copy()
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        img[:] = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
        cv2.putText(img, "Detection Area", (pts[0][0], max(pts[0][1] - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.line(img, (0, counting_line_y), (img.shape[1], counting_line_y), (0, 0, 255), 2)

def draw_info_panel(img, info_lines, origin=(10, 10)):
    wmax = 0
    for line in info_lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        wmax = max(wmax, tw)
    pad = 6
    line_h = 18
    box_w = wmax + pad * 2
    box_h = line_h * len(info_lines) + pad * 2
    x, y = origin
    rect_with_opacity(img, (x, y), (x + box_w, y + box_h), color=(0, 0, 0), alpha=0.5, border_thickness=2)
    for i, line in enumerate(info_lines):
        ly = y + pad + (i + 1) * line_h - 4
        cv2.putText(img, line, (x + pad, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def save_snapshot(base_frame, attrs, save_dir, filename_root):
    snap = base_frame.copy()
    draw_polygon_and_line(snap)
    info_lines = [
        f"Date: {attrs['start_date']} {attrs['start_time']}",
        f"Video: {attrs['video_time']} / {attrs['total_time']} (f{attrs['frame_index']}/{attrs['total_frames']})",
        f"Line: {attrs['line_mode']} y={attrs['counting_line_y']}",
        f"ROI pts: {attrs['roi_points_count']}",
        f"Min sizes: {size_thresholds}",
    ]
    draw_info_panel(snap, info_lines, origin=(10, 10))
    img_path = os.path.join(save_dir, f"{filename_root}.jpg")
    json_path = img_path.replace(".jpg", ".json")
    cv2.imwrite(img_path, snap)
    with open(json_path, 'w') as jf:
        json.dump(attrs, jf, indent=2)
    print(f"[SNAPSHOT] Saved: {img_path}")
    print(f"[SNAPSHOT] Meta  : {json_path}")

def crossed_counting_line_dir(prev_y, curr_y, line_y):
    if prev_y is None or curr_y is None: return 0
    if prev_y == curr_y: return 0
    prev_side = prev_y - line_y
    curr_side = curr_y - line_y
    crossed = (prev_side * curr_side) < 0 or (prev_side == 0 and curr_side != 0) or (curr_side == 0 and prev_side != 0)
    if not crossed: return 0
    return +1 if curr_y > prev_y else -1

def inside_button(x, y, rect):
    (x1, y1, x2, y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2

BTN_POLY  = (200, 20, 300, 60)
BTN_START = (320, 20, 420, 60)
BTN_LINE  = (440, 20, 540, 60)
BTN_EXIT  = (560, 20, 620, 60)

def ask_size_thresholds():
    """
    Prompts for start/end/capture times and per-class min sizes.
    Robust to user cancelling. Resets counters, folders, and logs safely.
    """
    global p1, p2, p3, p4
    global start_date, start_time, end_time, capture_time
    global log_file, summary_log_file
    global current_timestamp, previous_timestamp
    global counted_ids, track_info, tracking_trails
    global subfolder, filename_only

    # --- date/time prompts (keep old values if user cancels) ---
    new_p1 = simpledialog.askstring("Set Start Date", "Specify Date? (dd/mm/YYYY)", initialvalue=p1)
    if new_p1:
        try:
            start_date = datetime.strptime(new_p1, "%d/%m/%Y").date()
            p1 = new_p1
        except Exception as e:
            print("[WARN] Bad start date, keeping previous:", e)

    new_p2 = simpledialog.askstring("Set Start Time", "Specify Start Time? (HH:MM:SS)", initialvalue=p2)
    if new_p2:
        try:
            start_time = datetime.combine(start_date, datetime.strptime(new_p2, "%H:%M:%S").time())
            p2 = new_p2
        except Exception as e:
            print("[WARN] Bad start time, keeping previous:", e)

    new_p3 = simpledialog.askstring("Set End Time", "Specify End Time? (HH:MM:SS)", initialvalue=p3)
    if new_p3:
        try:
            end_time = datetime.combine(start_date, datetime.strptime(new_p3, "%H:%M:%S").time())
            p3 = new_p3
        except Exception as e:
            print("[WARN] Bad end time, keeping previous:", e)

    new_p4 = simpledialog.askstring("Set capture Time", "Specify Capture Time? (HH:MM:SS)", initialvalue=p4)
    if new_p4:
        try:
            capture_time = datetime.combine(start_date, datetime.strptime(new_p4, "%H:%M:%S").time())
            p4 = new_p4
        except Exception as e:
            print("[WARN] Bad capture time, keeping previous:", e)

    current_timestamp = start_time
    previous_timestamp = capture_time

    # --- per-class size thresholds ---
    for label in list(size_thresholds.keys()):
        try:
            w = simpledialog.askinteger("Set Min Width", f"Min width for {label}",
                                        initialvalue=size_thresholds[label][0], minvalue=0)
            h = simpledialog.askinteger("Set Min Height", f"Min height for {label}",
                                        initialvalue=size_thresholds[label][1], minvalue=0)
            if w is not None and h is not None:
                size_thresholds[label] = (w, h)
        except Exception as e:
            print(f"[WARN] Size prompt error for {label}:", e)

    # restart video from beginning (if file)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except Exception:
        pass

    # reset counts/state
    vehicle_count_by_class.clear()
    vehicle_count_by_group.clear()
    counted_ids.clear()
    track_info.clear()
    tracking_trails.clear()

    # reset output folder
    try:
        shutil.rmtree(subfolder)
    except Exception:
        pass
    os.makedirs(subfolder, exist_ok=True)

    # re-open logs
    try:
        log_file.close()
    except Exception:
        pass
    try:
        summary_log_file.close()
    except Exception:
        pass

    log_file_path = f"{filename_only}_log.csv"
    summary_file_path = f"{filename_only}_summary_log.csv"
    log_file = open(log_file_path, 'w', buffering=1)
    log_file.write("Timestamp,ID,Type,Confidence,VideoTime,Image\n")
    summary_log_file = open(summary_file_path, 'w', buffering=1)
    summary_log_file.write(f"{filename_only}\n")
    summary_log_file.write(f"{p1}\n")
    summary_log_file.write("Masa, motorcycle, pickup, van, lorry, bus, car\n")

def mouse_callback(event, x, y, flags, param):
    global polygon_points, polygon_defined, poly_editing, current_mouse
    global running, start_processing, do_capture, line_edit_mode, counting_line_y, line_mode
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if inside_button(x, y, BTN_POLY):
            polygon_points.clear()
            polygon_defined = False
            poly_editing = True
            print("[INFO] Polygon edit: left-click add points, right-click or 'p' to close.")
            return
        if inside_button(x, y, BTN_START):
            start_processing = True
            do_capture = True
            ask_size_thresholds()
            print("[INFO] Start clicked")
            return
        if inside_button(x, y, BTN_LINE):
            line_edit_mode = True
            print("[INFO] Line edit mode: click to set counting_line_y (press 'A' for AUTO).")
            return
        if inside_button(x, y, BTN_EXIT):
            running = False
            print("[INFO] Exit clicked")
            return
        if poly_editing:
            polygon_points.append((x, y))
            return
        if line_edit_mode:
            globals()['counting_line_y'] = y
            globals()['line_mode'] = 'MANUAL'
            globals()['line_edit_mode'] = False
            print(f"[INFO] Manual counting_line_y set to {counting_line_y}")
            return
    if event == cv2.EVENT_RBUTTONDOWN:
        if poly_editing and len(polygon_points) >= 3:
            polygon_defined = True
            poly_editing = False
            print(f"[INFO] Polygon defined with {len(polygon_points)} points")
            if 'frame_for_mask' in globals() and frame_for_mask is not None:
                rebuild_polygon_mask(frame_for_mask.shape)
            return

def classify_crop(crop_bgr):
    """Return (name, conf) from classifier; expects SAME 6 names."""
    if clf is None or crop_bgr is None or crop_bgr.size == 0:
        return None, None
    r = clf.predict(source=crop_bgr, imgsz=CLS_IMGSZ, device=DEVICE, verbose=False)[0]
    name = r.names[int(r.probs.top1)]
    conf = float(r.probs.top1conf)
    return name, conf

def restore_from_snapshot(attrs):
    global polygon_points, polygon_defined, poly_editing, polygon_mask
    global counting_line_y, line_mode, size_thresholds
    global start_date, p1, start_time, p2, end_time, p3, capture_time, p4
    sd = attrs.get("start_date")
    if sd:
        try:
            start_date = datetime.strptime(sd, "%d/%m/%Y").date() if "/" in sd else datetime.fromisoformat(sd).date()
            p1 = start_date.strftime("%d/%m/%Y")
        except Exception as e:
            print(f"[SNAPSHOT] start_date parse failed ({sd}): {e}")
    st = attrs.get("start_time")
    if st:
        try:
            fmt = "%H:%M:%S" if st.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(st, fmt).time()
            start_time = datetime.combine(start_date, t_val)
            p2 = start_time.strftime("%H:%M:%S")
        except Exception as e:
            print(f"[SNAPSHOT] start_time parse failed ({st}): {e}")
    et = attrs.get("end_time")
    if et:
        try:
            fmt = "%H:%M:%S" if et.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(et, fmt).time()
            end_time = datetime.combine(start_date, t_val)
            p3 = end_time.strftime("%H:%M:%S")
        except Exception as e:
            print(f"[SNAPSHOT] end_time parse failed ({et}): {e}")
    ct = attrs.get("capture_time")
    if ct:
        try:
            fmt = "%H:%M:%S" if ct.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(ct, fmt).time()
            capture_time = datetime.combine(start_date, t_val)
            p4 = capture_time.strftime("%H:%M:%S")
        except Exception as e:
            print(f"[SNAPSHOT] capture_time parse failed ({ct}): {e}")
    pts = attrs.get("roi_points") or []
    polygon_points = [tuple(map(int, p)) for p in pts]
    polygon_defined = len(polygon_points) >= 3
    poly_editing = False
    polygon_mask = None
    line_mode = attrs.get("line_mode", line_mode)
    try:
        counting_line_y = int(attrs.get("counting_line_y", counting_line_y))
    except Exception:
        pass
    st_map = attrs.get("size_thresholds") or {}
    for k, v in st_map.items():
        try:
            size_thresholds[k] = (int(v[0]), int(v[1]))
        except Exception:
            pass
    if line_mode == "AUTO" and polygon_defined:
        counting_line_y = polygon_mid_y(polygon_points)
    print("[SNAPSHOT] Restored:",
          f"ROI pts={len(polygon_points)} | line_mode={line_mode} y={counting_line_y} | thresholds={size_thresholds}")

def load_latest_snapshot(store_folder, filename_root=None):
    try:
        path = os.path.join(store_folder, f"{filename_root}.json")
        if not os.path.isfile(path):
            print(f"[SNAPSHOT] No json file found: {path}")
            return None, None
        with open(path, "r") as jf:
            data = json.load(jf)
        return path, data
    except Exception as e:
        print(f"[SNAPSHOT] Load error: {e}")
        return None, None

# Tk GUI
root = tk.Tk(); root.withdraw()

video_path = filedialog.askopenfilename(
    title="Select Video File or cancel to type URL",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    video_path = simpledialog.askstring("Video URL", "Enter camera URL (e.g. rtsp://user:pass@host:554/stream):")
    if not video_path:
        print("[ERROR] No source provided. Exiting.")
        raise SystemExit

filename_only = make_safe_dir_name(video_path)
try:
    os.makedirs(output_dir, exist_ok=True)
except PermissionError:
    output_dir = os.path.join(os.path.expanduser("~"), "vehicle_captures")
    os.makedirs(output_dir, exist_ok=True)
subfolder = os.path.join(output_dir, filename_only)
try: shutil.rmtree(subfolder)
except Exception: pass
os.makedirs(subfolder, exist_ok=True)
print(f"[IO] Saving crops & snapshots under: {os.path.abspath(output_dir)}")

GPU_INFO = gpu_availability_report()

snap_path, snap_attrs = load_latest_snapshot(output_dir, filename_only)
if snap_attrs:
    print(f"[SNAPSHOT] Loading: {snap_path}")
    restore_from_snapshot(snap_attrs)
    start_processing = True
else:
    print("[SNAPSHOT] No snapshot JSON found; starting with defaults.")

# Streaming config
STREAM_ENABLED = False
STREAM_URL = "rtsp://localhost:8554/"+filename_only
USE_NVENC = False
try:
    if (GPU_INFO.get("gpu_count", 0) > 0 or GPU_INFO.get("nvidia_smi")) and GPU_INFO.get("ffmpeg_nvenc"):
        USE_NVENC = True
        print("[GPU] Auto-enabled NVENC for FFmpeg streaming.")
except Exception:
    pass

cv2.namedWindow("Traffic Analysis")
cv2.setMouseCallback("Traffic Analysis", mouse_callback)

# RTSP low-latency flags
if str(video_path).startswith("rtsp://"):
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;tcp|max_delay;0|reorder_queue_size;0|stimeout;3000000"

cap = cv2.VideoCapture(video_path)
try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception: pass

back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"[INFO] fps:{fps} total_frames: {total_frames}")

log_file = open(f"{filename_only}_log.csv", 'w', buffering=1)
log_file.write("Timestamp,ID,Type,Confidence,VideoTime,Image\n")
summary_log_file = open(f"{filename_only}_summary_log.csv", 'w', buffering=1)
summary_log_file.write(f"{filename_only}\n")
summary_log_file.write(f"{p1}\n")
summary_log_file.write("Masa, motorcycle, pickup, van, lorry, bus, car\n")

current_timestamp = start_time
check_timestamp = capture_time
previous_timestamp = capture_time
print(f"[INFO] current_timestamp: {current_timestamp} previous_timestamp: {previous_timestamp}")
last_good_frame = None
frame_for_mask = None

streamer = None
stream_on = STREAM_ENABLED

read_failures = 0
t0_live = time.monotonic()
time_start = False
time_end = False

try:
    while running:
        ret, frame = cap.read()
        if (not ret) or time_end:
            if total_frames == 0:  # live source: keep trying
                read_failures += 1
                if read_failures % 100 == 0:
                    print("[CAP] Reopening live source...")
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception: pass
                time.sleep(0.02)
                if stream_on and streamer and streamer.proc is None:
                    streamer = None
                continue
            print("[INFO] COMPLETED")
            break
        else:
            read_failures = 0

        frame = cv2.resize(frame, (1280, 720))
        last_good_frame = frame.copy()
        frame_for_mask = frame

        if poly_editing and len(polygon_points) >= 2:
            pass
        elif polygon_defined and polygon_mask is not None:
            pass
        elif polygon_defined:
            rebuild_polygon_mask(frame.shape)

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if total_frames > 0 else 0

        if total_frames > 0:
            current_time_sec = current_frame / fps
            total_time_sec = total_frames / fps
        else:
            current_time_sec = time.monotonic() - t0_live
            total_time_sec = 0.0
        current_time_str = time.strftime("%H:%M:%S", time.gmtime(current_time_sec))
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time_sec)) if total_time_sec else "--:--:--"

        if total_frames > 0 and current_frame == total_frames:
            do_capture = True

        time_obj_local = datetime.strptime(current_time_str, "%H:%M:%S").time()
        delta = timedelta(hours=time_obj_local.hour, minutes=time_obj_local.minute, seconds=time_obj_local.second)
        result = start_time + delta
        if result > capture_time: time_start = True
        if result > end_time: time_end = True
        if int(time_obj_local.second) == 0 and int(time_obj_local.minute) > 0:
            if result.time() > current_timestamp.time():
                current_timestamp = result

        display = frame.copy()
        draw_polygon_and_line(display)

        draw_btn(display, BTN_POLY, "Poly", (100, 100, 0))
        draw_btn(display, BTN_START, "Start", (0, 200, 0))
        draw_btn(display, BTN_LINE, "Line", (200, 100, 0))
        draw_btn(display, BTN_EXIT, "Exit", (0, 0, 200))

        if start_processing and polygon_defined and time_start:
            fg_mask = back_sub.apply(frame)
            if polygon_mask is not None:
                fg_mask = cv2.bitwise_and(fg_mask, polygon_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.putText(fg_mask, f"{current_time_str} / {total_time_str}", (470, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

            # ---- YOLO detection (6 classes) ----
            with torch.inference_mode():
                res = det.predict(source=frame, imgsz=1280, conf=0.25, iou=0.45, device=DEVICE, verbose=False)[0]

            # DEBUG: print once what the result contains
            if DEBUG and not _printed_chk:
                print("[CHK] result has boxes:", hasattr(res, "boxes"), "len=", (len(res.boxes) if getattr(res, "boxes", None) is not None else "None"))
                print("[CHK] result has probs:", hasattr(res, "probs"))
                _printed_chk = True

            # SAFE: handle None/empty boxes
            boxes = getattr(res, "boxes", None)
            if not boxes or len(boxes) == 0:
                # No detections this frame (or wrong model type already caught earlier)
                cv2.imshow("Foreground Mask", fg_mask)
                cv2.imshow("Traffic Analysis", display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                elif key == ord('t'):
                    stream_on = not stream_on
                    if not stream_on and streamer:
                        streamer.close(); streamer = None
                    print(f"[STREAM] {'ON' if stream_on else 'OFF'}")
                continue

            detections = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                det_name = det.names[cls_id]  # detector label

                # Min-size filter (by detector label)
                if det_name in size_thresholds:
                    min_w, min_h = size_thresholds[det_name]
                    if w < min_w or h < min_h:
                        continue

                center = get_center(x1, y1, w, h)
                if not point_in_polygon(center):
                    continue

                # Crop for classifier
                crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]

                final_name = det_name
                final_conf = conf

                # refine with classifier (if enabled)
                if USE_CLASSIFIER and crop.size > 0 and (REFINE_ONLY is None or det_name in REFINE_ONLY):
                    cls_name, cls_conf = classify_crop(crop)
                    if cls_name:  # trust classifier trained on the same 6 names
                        final_name = cls_name
                        final_conf = cls_conf

                color_bgr = COLOR_MAP.get(final_name, (255,255,255))
                detections.append((x1, y1, x2, y2, final_name, final_conf, center, w, h, color_bgr))

            input_centroids = [det[6] for det in detections]
            objects = tracker.update(input_centroids)

            # associate tracks to detections
            track_to_det = {}
            if detections:
                det_centers = [det[6] for det in detections]
                unused = set(range(len(detections)))
                for objectID, trk_centroid in objects.items():
                    best_i, best_d = None, float("inf")
                    for i in list(unused):
                        dcx, dcy = det_centers[i]
                        d = (trk_centroid[0] - dcx)**2 + (trk_centroid[1] - dcy)**2
                        if d < best_d:
                            best_d, best_i = d, i
                    if best_i is not None:
                        track_to_det[objectID] = best_i
                        unused.discard(best_i)

            # clean stale tracks
            stale_ids = set(track_info.keys()) - set(objects.keys())
            for sid in stale_ids:
                del track_info[sid]
                tracking_trails.pop(sid, None)

            for objectID, trk_centroid in objects.items():
                if objectID not in track_info:
                    track_info[objectID] = {"last_centroid": None, "counted": False, "last_y_bottom": None}

                det_idx = track_to_det.get(objectID, None)
                if det_idx is None:
                    tracking_trails[objectID].append(trk_centroid)
                    for pt in tracking_trails[objectID]:
                        cv2.circle(display, pt, 2, (160, 160, 160), -1)
                    track_info[objectID]["last_centroid"] = trk_centroid
                    continue

                x1, y1, x2, y2, label, label_conf, center, w, h, color_bgr = detections[det_idx]
                prev_centroid = track_info[objectID]["last_centroid"]
                already_counted = track_info[objectID]["counted"]

                curr_y_bottom = y2
                prev_y_bottom = track_info[objectID].get("last_y_bottom")

                if not already_counted:
                    direction = crossed_counting_line_dir(prev_y=prev_y_bottom, curr_y=curr_y_bottom, line_y=counting_line_y)
                    if ((COUNT_DIR == 'down' and direction == +1) or
                        (COUNT_DIR == 'up' and direction == -1) or
                        (COUNT_DIR == 'both' and direction != 0)):

                        track_info[objectID]["counted"] = True
                        counted_ids.add(objectID)
                        vehicle_count_by_class[label] += 1
                        vehicle_count_by_group[label] += 1

                        # slightly padded crop for saving
                        pad = 10 if label == "pickup" else 5
                        cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
                        cx2 = min(frame.shape[1], x2 + pad); cy2 = min(frame.shape[0], y2 + pad)
                        cropped = frame[cy1:cy2, cx1:cx2]
                        filename = os.path.join(subfolder, f"{label}_{objectID}.jpg")
                        if cropped.size > 0:
                            cv2.imwrite(filename, cropped)
                            log_file.write(f"{current_timestamp.time()},{objectID},{label},{label_conf:.2f},{current_time_str},{filename}\n")
                            log_file.flush()
                            last_crop = cv2.resize(cropped, (120, 70))
                            last_label = f"{label} {w}x{h}"
                            print(f"saved: {label}_{objectID}.jpg @ {current_timestamp.time()} -> {last_label} total:{vehicle_count_by_class[label]}")

                        # periodic summary each minute
                        if (current_timestamp.time() > previous_timestamp.time()) or do_capture:
                            check_timestamp = previous_timestamp + timedelta(minutes=1)
                            order = ["motorcycle","pickup","van","lorry","bus","car"]
                            row = [vehicle_count_by_group.get(k,0) for k in order]
                            summary_log_file.write(f"{previous_timestamp.strftime('%H:%M')} - {check_timestamp.strftime('%H:%M')},{','.join(map(str,row))}\n")
                            summary_log_file.flush()
                            for key in list(vehicle_count_by_group.keys()):
                                vehicle_count_by_group[key] = 0
                            previous_timestamp = check_timestamp

                tracking_trails[objectID].append(trk_centroid)
                for pt in tracking_trails[objectID]:
                    cv2.circle(fg_mask, pt, 2, (160, 160, 160), -1)
                cv2.rectangle(display, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(display, f"{label}:{label_conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

                track_info[objectID]["last_centroid"] = trk_centroid
                track_info[objectID]["last_y_bottom"] = curr_y_bottom

            # counting line + stats
            cv2.line(display, (0, counting_line_y), (1280, counting_line_y), (0, 0, 255), 2)
            y_offset = 200
            for idx, (lbl, count) in enumerate(sorted(vehicle_count_by_class.items())):
                cv2.putText(display, f"{lbl}: {count}", (10, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            if last_crop is not None:
                x_offset, y_offset2 = 520, 290
                display[y_offset2:y_offset2 + last_crop.shape[0], x_offset:x_offset + last_crop.shape[1]] = last_crop
                cv2.rectangle(display, (x_offset, y_offset2 - 20), (x_offset + 120, y_offset2), (0, 0, 0), -1)
                cv2.putText(display, last_label, (x_offset, y_offset2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Foreground Mask", fg_mask)

        cv2.putText(display, f"'t'=Stream: {'ON' if stream_on else 'OFF'} -> {STREAM_URL if stream_on else ''}",
                    (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0) if stream_on else (0,0,255), 1)
        cv2.putText(display, f"Dir: d=down u=up b=both | Poly: L-add R-close p=close u=undo r=reset | Line: l edit a auto",
                    (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.putText(display, "General: c=capture, t=toggle stream, ESC=quit",
                    (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # STREAM OUT
        if stream_on:
            if streamer is None or streamer.proc is None:
                streamer = FFmpegStreamer(STREAM_URL, fps=fps, width=display.shape[1], height=display.shape[0], use_nvenc=USE_NVENC)
                print(f"[STREAM] Publishing to {STREAM_URL}")
            streamer.write(display)

        cv2.imshow("Traffic Analysis", display)

        if 'do_capture' in locals() and do_capture:
            attrs = {
                "video_name": filename_only,
                "frame_index": current_frame,
                "total_frames": total_frames,
                "video_time": current_time_str,
                "total_time": total_time_str,
                "start_date": start_date.strftime("%d/%m/%Y"),
                "start_time": start_time.strftime("%H:%M:%S"),
                "capture_time": capture_time.strftime("%H:%M:%S"),
                "end_time": end_time.strftime("%H:%M:%S"),
                "line_mode": line_mode,
                "counting_line_y": int(counting_line_y),
                "roi_points_count": len(polygon_points),
                "roi_points": [(int(x), int(y)) for x, y in polygon_points],
                "size_thresholds": {k: [int(v[0]), int(v[1])] for k, v in size_thresholds.items()}
            }
            save_snapshot(frame, attrs, output_dir, filename_only)
            do_capture = False

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('u'):
            if poly_editing and polygon_points:
                polygon_points.pop()
            else:
                COUNT_DIR = 'up'; print("[COUNT] Direction set to UP")
        elif key == ord('d'):
            COUNT_DIR = 'down'; print("[COUNT] Direction set to DOWN")
        elif key == ord('b'):
            COUNT_DIR = 'both'; print("[COUNT] Direction set to BOTH")
        elif key == ord('r'):
            polygon_points.clear()
            polygon_defined = False
            poly_editing = False
            polygon_mask = None
            if line_mode == 'AUTO':
                counting_line_y = 250
        elif key == ord('p') and poly_editing and len(polygon_points) >= 3:
            polygon_defined = True
            poly_editing = False
            rebuild_polygon_mask(frame.shape)
        elif key == ord('l'):
            line_edit_mode = True
            print("[INFO] Line edit mode: click anywhere to set counting_line_y (press 'A' for AUTO).")
        elif key == ord('a'):
            line_mode = 'AUTO'
            if polygon_defined:
                counting_line_y = polygon_mid_y(polygon_points)
            print(f"[INFO] Line mode set to AUTO (y={counting_line_y})")
        elif key == ord('c'):
            do_capture = True
        elif key == ord('t'):
            stream_on = not stream_on
            if not stream_on and streamer:
                streamer.close(); streamer = None
            print(f"[STREAM] {'ON' if stream_on else 'OFF'}")

except Exception:
    print("\n[ERROR] Uncaught exception. Traceback:")
    traceback.print_exc()

# Cleanup
try: log_file.close()
except: pass
try: summary_log_file.close()
except: pass
if streamer: streamer.close()
cap.release()
cv2.destroyAllWindows()
