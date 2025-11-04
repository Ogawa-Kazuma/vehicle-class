import cv2
import torch
import numpy as np
import time
import os, sys
import re
import json
import shutil
import subprocess
from urllib.parse import urlparse
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, simpledialog
from collections import defaultdict, OrderedDict
from ultralytics import YOLO
try:
    import torch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

# =========================
# Global perf switches
# =========================
cv2.setUseOptimized(True)
cv2.setNumThreads(0)               # let OpenCV decide threads
torch.backends.cudnn.benchmark = True  # autotune best cudnn algos

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
            'ffmpeg',
            '-re',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s:v', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',               # stdin
            '-an',
            '-c:v', vcodec,
        ]
        if use_nvenc:
            args += ['-preset', 'p3', '-tune', 'ull', '-rc', 'vbr', '-b:v', '2M', '-g', str(self.fps)]
        else:
            args += ['-preset', 'veryfast', '-tune', 'zerolatency', '-b:v', '2M', '-g', str(self.fps)]

        args += ['-pix_fmt', 'yuv420p']

        if out_format == 'rtsp':
            args += ['-f', 'rtsp', '-rtsp_transport', 'tcp', self.url]
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
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedCols = set(range(len(input_centroids))) - usedCols
            for col in unusedCols:
                self.register(input_centroids[col])

        return self.objects
def ffmpeg_has_nvenc():
    """Return True if ffmpeg exposes NVENC encoders."""
    try:
        out = subprocess.check_output(
            ['ffmpeg', '-hide_banner', '-encoders'],
            text=True, stderr=subprocess.STDOUT, timeout=4
        )
        return ('h264_nvenc' in out) or ('hevc_nvenc' in out)
    except Exception:
        return False

def gpu_availability_report():
    """
    Print a friendly GPU report:
    - PyTorch CUDA availability & build version
    - Detected GPUs (name, CC, memory) if visible to torch
    - nvidia-smi summary (if available)
    - FFmpeg NVENC presence
    Returns a dict with the collected info.
    """
    info = {
        "torch_cuda_available": None,
        "torch_cuda_version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_caps": [],
        "gpu_mem_gb": [],
        "nvidia_smi": None
    }

    # PyTorch view
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

    # nvidia-smi (works even if CUDA is hidden from torch)
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'],
            text=True, stderr=subprocess.STDOUT, timeout=3
        )
        lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
        info["nvidia_smi"] = lines
    except Exception as e:
        info["nvidia_smi_error"] = str(e)

    # FFmpeg NVENC
    info["ffmpeg_nvenc"] = ffmpeg_has_nvenc()

    # ---- pretty print ----
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
    import sys, os, torch
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _ = torch.cuda.get_device_name(0)  # harmless probe
            return "cuda:0"
    except Exception as e:
        print(f"[CUDA] Falling back to CPU: {e}", file=sys.stderr)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # hide CUDA for this process
    return "cpu"

DEVICE = select_device()
print(f"[DEVICE] Using {DEVICE}")
# =========================
# YOLOv8
# =========================
model = YOLO('yolov8s.pt')

# =========================
# Globals
# =========================
running = True
start_processing = False

# ROI polygon
polygon_points = []
polygon_defined = False
poly_editing = False
polygon_mask = None
current_mouse = (0, 0)

# Counting line
counting_line_y = 250
line_mode = 'AUTO'
line_edit_mode = False

# Directional counting (default: down)
COUNT_DIR = 'down'   # 'down' (default), 'up', 'both'

vehicle_count_by_class = defaultdict(int)
vehicle_count_by_group = defaultdict(int)

# Output directory (built later with safe name)
output_dir = "vehicle_captures"

counted_ids = set()
tracking_trails = defaultdict(list)
tracker = CentroidTracker(max_distance=90)
last_crop = None
last_label = ""
do_capture = False

# Robust counting
track_info = {}   # objectID -> {"last_centroid": (x,y) or None, "counted": False, "last_y_bottom": None}

# Time setup
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
# Size thresholds (min width, min height)
size_thresholds = {
    'motorcycle': (5, 10),
    'car': (30, 50),
    'bus': (60, 100),
    'truck': (40, 90),
}

# =========================
# Helpers
# =========================
def make_safe_dir_name(source: str) -> str:
    """Return a filesystem-safe name derived from a file path or URL."""
    if "://" in source:  # e.g., rtsp/rtmp/http
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
def rect_with_opacity(img, pt1, pt2, color=(0, 0, 0), alpha=0.5, border_thickness=0, border_color=(255,255,255)):
    """
    Draw a filled rectangle with opacity, optionally add a solid border.
    - img: BGR image (numpy array)
    - pt1, pt2: (x, y) top-left and bottom-right
    - color: BGR fill color
    - alpha: 0.0..1.0 (fill opacity)
    - border_thickness: >0 to draw a border on top (opaque)
    """
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)            # filled on overlay
    # Blend overlay -> img in place
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)

    if border_thickness > 0:
        cv2.rectangle(img, pt1, pt2, border_color, thickness=border_thickness)

    return img
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
    #cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    rect_with_opacity(img, (x, y), (x + box_w, y + box_h), color=(0, 0, 0), alpha=0.5, border_thickness=2)
    #cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (255, 255, 255), 1)
    
    for i, line in enumerate(info_lines):
        ly = y + pad + (i + 1) * line_h - 4
        cv2.putText(img, line, (x + pad, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def save_snapshot(base_frame, attrs, save_dir, filename_root):
    snap = base_frame.copy()
    draw_polygon_and_line(snap)  # no buttons drawn
    info_lines = [
        f"Date: {attrs['start_date']} {attrs['start_time']}",
        f"Video: {attrs['video_time']} / {attrs['total_time']} (f{attrs['frame_index']}/{attrs['total_frames']})",
        f"Line: {attrs['line_mode']} y={attrs['counting_line_y']}",
        f"ROI pts: {attrs['roi_points_count']}",
        f"Min sizes: mc={size_thresholds.get('motorcycle', (0,0))}, car={size_thresholds.get('car',(0,0))}",
        f"           bus={size_thresholds.get('bus',(0,0))}, truck={size_thresholds.get('truck',(0,0))}"
    ]
    draw_info_panel(snap, info_lines, origin=(10, 10))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(save_dir, f"{filename_root}.jpg")
    json_path = img_path.replace(".jpg", ".json")
    cv2.imwrite(img_path, snap)
    with open(json_path, 'w') as jf:
        json.dump(attrs, jf, indent=2)
    print(f"[SNAPSHOT] Saved: {img_path}")
    print(f"[SNAPSHOT] Meta  : {json_path}")

# Direction-aware crossing (uses only counting_line_y)
def crossed_counting_line_dir(prev_y, curr_y, line_y):
    """
    +1  => crossed downward  (moving to larger y)
    -1  => crossed upward    (moving to smaller y)
     0  => no crossing
    """
    if prev_y is None or curr_y is None:
        return 0
    if prev_y == curr_y:
        return 0

    prev_side = prev_y - line_y
    curr_side = curr_y - line_y

    crossed = (prev_side * curr_side) < 0 \
              or (prev_side == 0 and curr_side != 0) \
              or (curr_side == 0 and prev_side != 0)
    if not crossed:
        return 0

    return +1 if curr_y > prev_y else -1

# UI areas
def inside_button(x, y, rect):
    (x1, y1, x2, y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2

BTN_POLY    = (200, 20, 300, 60)
BTN_START   = (320, 20, 420, 60)
BTN_LINE    = (440, 20, 540, 60)
BTN_EXIT    = (560, 20, 620, 60)

def ask_size_thresholds():
    global p1,p2,p3,p4
    global start_date, start_time, end_time, capture_time
    global log_file, summary_log_file
    global current_timestamp, previous_timestamp
    global counted_ids, track_info, tracking_trails
    
    p1 = simpledialog.askstring("Set Start Date", "Specify Date? (dd/mm/YYYY)", initialvalue=p1)
    start_date = datetime.strptime(p1, "%d/%m/%Y").date()
    
    p2 = simpledialog.askstring("Set Start Time", "Specify Start Time? (HH:MM:SS)", initialvalue=p2)
    time_obj_local = datetime.strptime(p2, "%H:%M:%S").time()
    start_time = datetime.combine(start_date, time_obj_local)

    p3 = simpledialog.askstring("Set End Time", "Specify End Time? (HH:MM:SS)", initialvalue=p3)
    end_obj_local = datetime.strptime(p3, "%H:%M:%S").time()
    end_time = datetime.combine(start_date, end_obj_local)
    
    p4 = simpledialog.askstring("Set capture Time", "Specify Capture Time? (HH:MM:SS)", initialvalue=p4)
    capture_obj_local = datetime.strptime(p4, "%H:%M:%S").time()
    capture_time = datetime.combine(start_date, capture_obj_local)
    
    print(f"{capture_time}")
    current_timestamp = start_time
    previous_timestamp = capture_time

    for label in size_thresholds:
        w = simpledialog.askinteger("Set Min Width", f"Min width for {label}", initialvalue=size_thresholds[label][0], minvalue=0)
        h = simpledialog.askinteger("Set Min Height", f"Min height for {label}", initialvalue=size_thresholds[label][1], minvalue=0)
        size_thresholds[label] = (w or 0, h or 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Reset counts/state
    for key in list(vehicle_count_by_class.keys()):
        vehicle_count_by_class[key] = 0
    for key in list(vehicle_count_by_group.keys()):
        vehicle_count_by_group[key] = 0
    counted_ids = set()
    track_info = {}
    tracking_trails = defaultdict(list)
    
    # remove completely
    shutil.rmtree(subfolder)

    # recreate empty folder
    os.makedirs(subfolder, exist_ok=True)

    # Reopen logs
    try: log_file.close()
    except: pass
    try: summary_log_file.close()
    except: pass

    log_file_path = f"{filename_only}_log.csv"
    summary_file_path = f"{filename_only}_summary_log.csv"

    log_file = open(log_file_path, 'w', buffering=1)
    log_file.write("Timestamp,ID,Type,Confidence,VideoTime,Image\n")

    summary_log_file = open(summary_file_path, 'w', buffering=1)
    summary_log_file.write(f"{filename_only}\n")
    summary_log_file.write(f"{p1}\n")
    summary_log_file.write("Masa, Kelas 1, Kelas 2, Kelas 3, Kelas 4, Kelas 5, Kelas 6\n")

def mouse_callback(event, x, y, flags, param):
    global polygon_points, polygon_defined, poly_editing, current_mouse
    global running, start_processing, do_capture, line_edit_mode, counting_line_y, line_mode

    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        #if inside_button(x, y, BTN_CAPTURE):
        #    do_capture = True
        #    return
        if inside_button(x, y, BTN_POLY):
            polygon_points = []
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
            print("[INFO] Line edit mode: click anywhere to set counting_line_y (press 'A' for AUTO).")
            return
        if inside_button(x, y, BTN_EXIT):
            running = False
            print("[INFO] Exit clicked")
            return

        if poly_editing:
            polygon_points.append((x, y))
            return

        if line_edit_mode:
            counting_line_y = y
            line_mode = 'MANUAL'
            line_edit_mode = False
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
def restore_from_snapshot(attrs):
    """Apply start_date, start_time, ROI polygon, counting-line settings, and size thresholds."""
    global polygon_points, polygon_defined, poly_editing, polygon_mask
    global counting_line_y, line_mode, size_thresholds
    global start_date, p1, start_time, p2, end_time, p3, capture_time, p4  # <- include start_time + p2

    # --- start_date (string in snapshot) ---
    sd = attrs.get("start_date")
    if sd:
        try:
            if "/" in sd:
                start_date = datetime.strptime(sd, "%d/%m/%Y").date()
            else:
                start_date = datetime.fromisoformat(sd).date()
            p1 = start_date.strftime("%d/%m/%Y")
            print(f"[SNAPSHOT] start_date restored: {p1}")
        except Exception as e:
            print(f"[SNAPSHOT] start_date parse failed ({sd}): {e}")

    # --- start_time (string "HH:MM" or "HH:MM:SS") ---
    st = attrs.get("start_time")
    if st:
        try:
            fmt = "%H:%M:%S" if st.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(st, fmt).time()
            # combine with (possibly restored) start_date
            start_time = datetime.combine(start_date, t_val)
            p2 = start_time.strftime("%H:%M:%S")
            print(f"[SNAPSHOT] start_time restored: {start_time}")
        except Exception as e:
            print(f"[SNAPSHOT] start_time parse failed ({st}): {e}")
    # --- end_time (string "HH:MM" or "HH:MM:SS") ---
    et = attrs.get("end_time")
    if et:
        try:
            fmt = "%H:%M:%S" if et.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(et, fmt).time()
            # combine with (possibly restored) start_date
            end_time = datetime.combine(start_date, t_val)
            p3 = end_time.strftime("%H:%M:%S")
            print(f"[SNAPSHOT] end_time restored: {end_time}")
        except Exception as e:
            print(f"[SNAPSHOT] end_time parse failed ({et}): {e}")
            # --- end_time (string "HH:MM" or "HH:MM:SS") ---
    ct = attrs.get("capture_time")
    if ct:
        try:
            fmt = "%H:%M:%S" if ct.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(ct, fmt).time()
            # combine with (possibly restored) start_date
            capture_time = datetime.combine(start_date, t_val)
            p4 = capture_time.strftime("%H:%M:%S")
            print(f"[SNAPSHOT] capture_time restored: {capture_time}")
        except Exception as e:
            print(f"[SNAPSHOT] capture_time parse failed ({ct}): {e}")
    # --- ROI polygon ---
    pts = attrs.get("roi_points") or []
    polygon_points = [tuple(map(int, p)) for p in pts]
    polygon_defined = len(polygon_points) >= 3
    poly_editing = False
    polygon_mask = None  # will be rebuilt on first frame

    # --- Counting line/mode ---
    line_mode = attrs.get("line_mode", line_mode)
    try:
        counting_line_y = int(attrs.get("counting_line_y", counting_line_y))
    except Exception:
        pass

    # --- Size thresholds ---
    st_map = attrs.get("size_thresholds") or {}
    for k, v in st_map.items():
        try:
            w, h = int(v[0]), int(v[1])
            size_thresholds[k] = (w, h)
        except Exception:
            pass

    # If AUTO and we have a polygon, derive midline from polygon bbox
    if line_mode == "AUTO" and polygon_defined:
        counting_line_y = polygon_mid_y(polygon_points)

    print("[SNAPSHOT] Restored:",
          f"ROI pts={len(polygon_points)} | line_mode={line_mode} y={counting_line_y} | thresholds={size_thresholds}")

def load_latest_snapshot(store_folder, filename_root=None):
    """
    Find the most recent *_snapshot_*.json under store_folder using filename-only
    heuristics (no filesystem mtime).
    """
    try:
        files = [f for f in os.listdir(store_folder) if f.endswith(".json")]
        if not files:
            return None, None
        
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

# Tk GUI bootstrap
root = tk.Tk()
root.withdraw()

# Input source (file OR RTSP URL)
video_path = filedialog.askopenfilename(title="Select Video File or cancel to type URL",
                                        filetypes=[["Video files", "*.mp4 *.avi *.mov *.mkv"]])
if not video_path:
    video_path = simpledialog.askstring("Video URL", "Enter camera URL (e.g. rtsp://user:pass@host:554/stream):")
    if not video_path:
        print("[ERROR] No source provided. Exiting.")
        raise SystemExit

# ---- Build safe output locations ----
filename_only = make_safe_dir_name(video_path)
try:
    os.makedirs(output_dir, exist_ok=True)
except PermissionError:
    output_dir = os.path.join(os.path.expanduser("~"), "vehicle_captures")
    os.makedirs(output_dir, exist_ok=True)
subfolder = os.path.join(output_dir, filename_only)
# remove completely
try:
    shutil.rmtree(subfolder)
except Exception:
    print("[INFO] folder not exist")
# recreate empty folder
os.makedirs(subfolder, exist_ok=True)
print(f"[IO] Saving crops & snapshots under: {os.path.abspath(output_dir)}")

GPU_INFO = gpu_availability_report()
# ---- Try to preload settings from latest snapshot JSON ----
snap_path, snap_attrs = load_latest_snapshot(output_dir, filename_only)
if snap_attrs:
    print(f"[SNAPSHOT] Loading: {snap_path}")
    restore_from_snapshot(snap_attrs)
    start_processing = True
else:
    print("[SNAPSHOT] No snapshot JSON found; starting with defaults.")

# =========================
# Streaming configuration
# =========================
STREAM_ENABLED = False  # set True to auto-start streaming, or press 't'
STREAM_URL = "rtsp://mywtpc.com:8554/"+filename_only  # RTSP (MediaMTX) example
# STREAM_URL = "rtmp://localhost/live/traffic"  # RTMP example
USE_NVENC = True  # True if you have NVIDIA GPU + ffmpeg with h264_nvenc
try:
    if (GPU_INFO.get("gpu_count", 0) > 0 or GPU_INFO.get("nvidia_smi")) and GPU_INFO.get("ffmpeg_nvenc"):
        # Only flip if you didn’t explicitly set USE_NVENC earlier
        USE_NVENC = True
        print("[GPU] Auto-enabled NVENC for FFmpeg streaming.")
except Exception:
    pass
cv2.namedWindow("Traffic Analysis")
cv2.setMouseCallback("Traffic Analysis", mouse_callback)

# ---- Low-latency capture options for RTSP (set BEFORE opening cap) ----
if str(video_path).startswith("rtsp://"):
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;tcp|max_delay;0|reorder_queue_size;0|stimeout;3000000"

cap = cv2.VideoCapture(video_path)  # uses FFmpeg backend typically
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"[INFO] fps:{fps} total_frames: {total_frames}")

log_file = open(f"{filename_only}_log.csv", 'w', buffering=1)
log_file.write("Timestamp,ID,Type,Confidence,VideoTime,Image\n")
summary_log_file = open(f"{filename_only}_summary_log.csv", 'w', buffering=1)
summary_log_file.write(f"{filename_only}\n")
summary_log_file.write(f"{p1}\n")
summary_log_file.write("Masa, Kelas 1, Kelas 2, Kelas 3, Kelas 4, Kelas 5, Kelas 6\n")
    
current_timestamp = start_time
check_timestamp = capture_time
previous_timestamp = capture_time 
print(f"[INFO] current_timestamp: {current_timestamp} previous_timestamp: {previous_timestamp}")
last_good_frame = None
frame_for_mask = None

# Streaming runtime objects
streamer = None
stream_on = STREAM_ENABLED

# Live-source resilience
read_failures = 0
t0_live = time.monotonic()
time_start = False
time_end = False

while running:
    ret, frame = cap.read()
    if (not ret) or time_end:
        # For live sources, attempt graceful reconnect
        if total_frames == 0:
            read_failures += 1
            if read_failures % 100 == 0:
                print("[CAP] Reopening live source...")
                cap.release()
                cap = cv2.VideoCapture(video_path)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
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

    # Time labels (file or live)
    if total_frames > 0:
        current_time_sec = current_frame / fps
        total_time_sec = total_frames / fps
    else:
        current_time_sec = time.monotonic() - t0_live
        total_time_sec = 0.0
    current_time_str = time.strftime("%H:%M:%S", time.gmtime(current_time_sec))
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time_sec)) if total_time_sec else "--:--:--"

    # Final frame snapshot for files
    if total_frames > 0 and current_frame == total_frames:
        do_capture = True

    time_obj_local = datetime.strptime(current_time_str, "%H:%M:%S").time()
    delta = timedelta(hours=time_obj_local.hour, minutes=time_obj_local.minute, seconds=time_obj_local.second)
    result = start_time + delta
    #print(f"{time_start} {capture_time}={result}")
    if result > capture_time:
        time_start = True
    if result > end_time:
        time_end = True
        
    if int(time_obj_local.second) == 0 and int(time_obj_local.minute) > 0:
        if result.time() > current_timestamp.time():
            current_timestamp = result
            #print(f"[INFO] {current_timestamp}")
            #print(f"{time_start} {capture_time}={current_timestamp}")

    display = frame.copy()
    draw_polygon_and_line(display)

    # Draw buttons
    #draw_btn(display, BTN_CAPTURE, "Capture", (60, 60, 60))
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

        # ---- YOLOv8 inference (car=2, motorcycle=3, bus=5, truck=7) ----
        #with torch.inference_mode():
        #    res = model(frame, classes=[2, 3, 5, 7], conf=0.25, iou=0.45, device=device, verbose=False)[0]
        with torch.inference_mode():
            res = model(frame, classes=[2,3,5,7], conf=0.25, iou=0.45, device=DEVICE, verbose=False)[0]
    
        detections = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            coco_name = model.names[cls_id]
            if coco_name in size_thresholds:
                min_w, min_h = size_thresholds[coco_name]
                if w < min_w or h < min_h:
                    continue
            center = get_center(x1, y1, w, h)
            if not point_in_polygon(center):
                continue
            if cls_id == 2:
                label, color_rgb = "Class 1", (255, 255, 255)
            elif cls_id == 3:
                label, color_rgb = "Class 6", (255, 255, 0)
            elif cls_id == 5:
                label, color_rgb = "Class 5", (0, 0, 255)
            elif cls_id == 7:
                label, color_rgb = "Class 3", (255, 0, 0)
            else:
                label, color_rgb = "Class 2", (0, 0, 0)
            color_bgr = color_rgb[::-1]
            detections.append((x1, y1, x2, y2, label, conf, center, w, h, color_bgr))

        input_centroids = [det[6] for det in detections]
        objects = tracker.update(input_centroids)

        # Associate tracks to detections
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

        # Clean up old track_info entries (those removed by tracker)
        stale_ids = set(track_info.keys()) - set(objects.keys())
        for sid in stale_ids:
            del track_info[sid]
            tracking_trails.pop(sid, None)

        for objectID, trk_centroid in objects.items():
            if objectID not in track_info:
                track_info[objectID] = {"last_centroid": None, "counted": False, "last_y_bottom": None}

            det_idx = track_to_det.get(objectID, None)
            if det_idx is None:
                #print(f"[INFO] Tracking {objectID}")
                cv2.putText(fg_mask, f"Tracking {objectID} {label}", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
                tracking_trails[objectID].append(trk_centroid)
                for pt in tracking_trails[objectID]:
                    cv2.circle(display, pt, 2, (160, 160, 160), -1)
                track_info[objectID]["last_centroid"] = trk_centroid
                # keep last_y_bottom unchanged when unmatched
                continue

            x1, y1, x2, y2, label, conf, center, w, h, color_bgr = detections[det_idx]
            prev_centroid = track_info[objectID]["last_centroid"]
            already_counted = track_info[objectID]["counted"]

            # --- Direction-aware crossing using bottom of the bbox ---
            curr_y_bottom = y2
            prev_y_bottom = track_info[objectID].get("last_y_bottom")

            if not already_counted:
                direction = crossed_counting_line_dir(
                    prev_y=prev_y_bottom,
                    curr_y=curr_y_bottom,
                    line_y=counting_line_y
                )

                if (
                    (COUNT_DIR == 'down' and direction == +1) or
                    (COUNT_DIR == 'up'   and direction == -1) or
                    (COUNT_DIR == 'both' and direction != 0)
                ):
                    track_info[objectID]["counted"] = True
                    counted_ids.add(objectID)
                    vehicle_count_by_class[label] += 1
                    vehicle_count_by_group[label] += 1
                    #print(f"{label} {vehicle_count_by_group[label]}")
                    if label == "Class 6":
                        x1-=10
                        x2+=10
                        y1-=10
                        y2+=10
                    else:
                        x1-=5
                        x2+=5
                        y1-=5
                        y2+=5
                        
                    h_frame, w_frame = frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w_frame, x2)
                    y2 = min(h_frame, y2)

                    if x2 > x1 and y2 > y1:
                        cropped = frame[y1:y2, x1:x2]
                        filename = os.path.join(subfolder, f"{label}_{objectID}.jpg")
                        cv2.imwrite(filename, cropped)
                        log_file.write(f"{current_timestamp.time()},{objectID},{label},{conf:.2f},{current_time_str},{filename}\n")
                        log_file.flush()
                        last_crop = cv2.resize(cropped, (120, 70))
                        last_label = f"{label} {w}x{h}"
                        print(f"filename: {label}_{objectID}.jpg {current_timestamp.time()} {last_label} {vehicle_count_by_group[label]}")
                    
                    if (current_timestamp.time() > previous_timestamp.time()) or do_capture:
                        check_timestamp = check_timestamp + timedelta(minutes=1)
                        kelas1 = vehicle_count_by_group["Class 1"]; kelas2 = 0
                        kelas3 = vehicle_count_by_group["Class 3"]; kelas4 = 0
                        kelas5 = vehicle_count_by_group["Class 5"]
                        kelas6 = vehicle_count_by_group["Class 6"]

                        summary_log_file.write(f"{previous_timestamp.strftime('%H:%M')} - {check_timestamp.strftime('%H:%M')},"
                                               f"{kelas1},{kelas2},{kelas3},{kelas4},{kelas5},{kelas6}\n")
                        summary_log_file.flush()
                        
                        for key in list(vehicle_count_by_group.keys()):
                            vehicle_count_by_group[key] = 0
                        previous_timestamp = previous_timestamp + timedelta(minutes=1)
                        print(f"[INFO] {current_timestamp}")            

            # Draw trail and bbox
            tracking_trails[objectID].append(trk_centroid)
            for pt in tracking_trails[objectID]:
                cv2.circle(fg_mask, pt, 2, (160, 160, 160), -1)
            cv2.rectangle(display, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(display, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 2)

            # remember for next frame
            track_info[objectID]["last_centroid"] = trk_centroid
            track_info[objectID]["last_y_bottom"] = curr_y_bottom

        # Counting line and stats
        cv2.line(display, (0, counting_line_y), (1280, counting_line_y), (0, 0, 255), 2)
        y_offset = 200
        for idx, (label, count) in enumerate(vehicle_count_by_class.items()):
            cv2.putText(display, f"{label}: {count}", (10, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if last_crop is not None:
            x_offset, y_offset = 520, 290
            display[y_offset:y_offset + last_crop.shape[0], x_offset:x_offset + last_crop.shape[1]] = last_crop
            cv2.rectangle(display, (x_offset, y_offset - 20), (x_offset + 120, y_offset), (0, 0, 0), -1)
            cv2.putText(display, last_label, (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Foreground Mask", fg_mask)

    cv2.putText(display, f"'t'=Stream: {'ON' if stream_on else 'OFF'} -> {STREAM_URL if stream_on else ''}",
                (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0) if stream_on else (0,0,255), 1)

    # ---------------- HUD: direction + streaming + help ----------------
    cv2.putText(display, f"Dir keys: d=down, u=up, b=both  |  (u=undo while polygon editing) {COUNT_DIR.upper()}",
                (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(display, "Poly: L-click add, R-click p=close u=undo r=reset",
                (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(display, "Line: l=line edit a=auto | General: c=capture, t=toggle stream",
                (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ---- STREAM OUT (send annotated frame) ----
    if stream_on:
        if streamer is None or streamer.proc is None:
            streamer = FFmpegStreamer(STREAM_URL, fps=fps, width=display.shape[1], height=display.shape[0], use_nvenc=USE_NVENC)
            print(f"[STREAM] Publishing to {STREAM_URL}")
        streamer.write(display)

    # Local preview
    cv2.imshow("Traffic Analysis", display)

    # Snapshot (no buttons on saved image)
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

    key = cv2.waitKey(int(1000/fps)) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('u'):
        if poly_editing and polygon_points:
            # undo last polygon point while editing
            polygon_points.pop()
        else:
            # set counting direction to UP when not editing
            COUNT_DIR = 'up'
            print("[COUNT] Direction set to UP")
    elif key == ord('d'):
        COUNT_DIR = 'down'
        print("[COUNT] Direction set to DOWN")
    elif key == ord('b'):
        COUNT_DIR = 'both'
        print("[COUNT] Direction set to BOTH")
    elif key == ord('r'):
        polygon_points = []
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
            streamer.close()
            streamer = None
        print(f"[STREAM] {'ON' if stream_on else 'OFF'}")

# Cleanup
try: log_file.close()
except: pass
try: summary_log_file.close()
except: pass
if streamer:
    streamer.close()
cap.release()
cv2.destroyAllWindows()
