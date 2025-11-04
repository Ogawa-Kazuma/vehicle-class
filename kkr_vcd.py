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

# =========================
# Configurable defaults (can be overridden by JSON/YAML config)
# =========================
YOLO_CONF = 0.35      # Confidence threshold
YOLO_IOU  = 0.50      # IoU threshold for NMS

USE_YOLO_TRACKER   = True   # Use Ultralytics ByteTrack; else use CentroidTracker
TRACK_MAX_DIST     = 50     # pixels (CentroidTracker association gate)
TRACK_MAX_DISAPPEAR= 30     # frames before track is forgotten

LINE_DEBOUNCE_PX   = 5      # pixels beyond the line after crossing
TRAIL_LEN          = 60     # trail points per object

SUMMARY_BUCKET_MINUTES = 1  # minutes per summary row

STREAM_ENABLED = False
STREAM_URL     = "rtsp://mywtpc.com:8554/{filename_only}"
USE_NVENC      = False

# =========================
# CONFIG LOADER (JSON/YAML)
# =========================
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

CONFIG_CANDIDATES = [
    "traffic_analyzer_config.yaml",
    "traffic_analyzer_config.yml",
    "traffic_analyzer_config.json",
]

SECTION_MAP = {
    ("yolo", "conf"): "YOLO_CONF",
    ("yolo", "iou"):  "YOLO_IOU",
    ("tracker", "use_yolo_tracker"):   "USE_YOLO_TRACKER",
    ("tracker", "max_dist"):           "TRACK_MAX_DIST",
    ("tracker", "max_disappear"):      "TRACK_MAX_DISAPPEAR",
    ("crossing", "debounce_px"):       "LINE_DEBOUNCE_PX",
    ("trail", "length"):               "TRAIL_LEN",
    ("summary", "bucket_minutes"):      "SUMMARY_BUCKET_MINUTES",
    ("stream", "enabled"):              "STREAM_ENABLED",
    ("stream", "url"):                  "STREAM_URL",
    ("stream", "use_nvenc"):            "USE_NVENC",
}

def _find_config_path():
    here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    candidates = [os.path.join(here, c) for c in CONFIG_CANDIDATES] + [os.path.join(os.getcwd(), c) for c in CONFIG_CANDIDATES]
    seen = set()
    ordered = [p for p in candidates if not (p in seen or seen.add(p))]
    for p in ordered:
        if os.path.isfile(p):
            return p
    return None

def load_config():
    path = _find_config_path()
    if not path:
        return {}
    try:
        with open(path, 'r') as f:
            if path.endswith(('.yaml', '.yml')) and yaml is not None:
                cfg = yaml.safe_load(f) or {}
            else:
                cfg = json.load(f)
        print(f"[CONFIG] Loaded: {path}")
        return cfg or {}
    except Exception as e:
        print(f"[CONFIG] Failed to load {path}: {e}")
        return {}

def _set_if_present(name, value):
    if value is None:
        return
    if name in globals():
        globals()[name] = value
        print(f"[CONFIG] {name} := {value}")

def apply_config(cfg, filename_only_value=None):
    if not cfg:
        return
    # flat keys
    for k, v in cfg.items():
        if isinstance(v, (str, int, float, bool)) and k in globals():
            _set_if_present(k, v)
    # sectioned keys
    for (sect, key), const_name in SECTION_MAP.items():
        if isinstance(cfg.get(sect), dict) and key in cfg[sect]:
            _set_if_present(const_name, cfg[sect][key])
    # stream URL substitution
    if filename_only_value is not None and isinstance(globals().get('STREAM_URL'), str):
        try:
            globals()['STREAM_URL'] = globals()['STREAM_URL'].replace('{filename_only}', str(filename_only_value))
        except Exception:
            pass

# Load config once (before we know filename_only)
cfg = load_config()
apply_config(cfg)

# =========================
# Global perf switches
# =========================
cv2.setUseOptimized(True)
try:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

# =========================
# Utilities
# =========================
def ffmpeg_has_nvenc():
    try:
        out = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders'], text=True, stderr=subprocess.STDOUT, timeout=4)
        return ('h264_nvenc' in out) or ('hevc_nvenc' in out)
    except Exception:
        return False

def gpu_availability_report():
    info = {"torch_cuda_available": None, "torch_cuda_version": None, "gpu_count": 0, "gpu_names": [], "gpu_caps": [], "gpu_mem_gb": [], "nvidia_smi": None}
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
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT, timeout=3)
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

# =========================
# Streaming
# =========================
class FFmpegStreamer:
    def __init__(self, url, fps, width, height, use_nvenc=False):
        self.url = url
        self.proc = None
        self.width = width
        self.height = height
        self.fps = int(round(fps)) if fps else 25
        vcodec = 'h264_nvenc' if use_nvenc else 'libx264'
        out_format = 'rtsp' if url.startswith('rtsp://') else 'flv'
        args = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-re', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s:v', f'{self.width}x{self.height}', '-r', str(self.fps),
            '-i', '-', '-an', '-c:v', vcodec,
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
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

# =========================
# Improved Centroid Tracker
# =========================
class CentroidTracker:
    def __init__(self, max_distance=50, max_disappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

        # Note: for production work, consider matching with Hungarian/linear sum assignment

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def update(self, input_centroids):
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        if len(input_centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        D = np.linalg.norm(np.array(objectCentroids)[:, None] - np.array(input_centroids)[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        usedRows, usedCols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_distance:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = input_centroids[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(len(objectCentroids))) - usedRows
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.max_disappeared:
                self.deregister(objectID)

        unusedCols = set(range(len(input_centroids))) - usedCols
        for col in unusedCols:
            self.register(input_centroids[col])

        return self.objects

# =========================
# Helper drawing & geometry
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

def rect_with_opacity(img, pt1, pt2, color=(0, 0, 0), alpha=0.5, border_thickness=0, border_color=(255,255,255)):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, thickness=-1)
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
        f"Min sizes: mc={size_thresholds.get('motorcycle', (0,0))}, car={size_thresholds.get('car',(0,0))}",
        f"           bus={size_thresholds.get('bus',(0,0))}, truck={size_thresholds.get('truck',(0,0))}"
    ]
    draw_info_panel(snap, info_lines, origin=(10, 10))
    img_path = os.path.join(save_dir, f"{filename_root}.jpg")
    json_path = img_path.replace(".jpg", ".json")
    cv2.imwrite(img_path, snap)
    with open(json_path, 'w') as jf:
        json.dump(attrs, jf, indent=2)
    print(f"[SNAPSHOT] Saved: {img_path}")
    print(f"[SNAPSHOT] Meta  : {json_path}")

# Direction-aware crossing with debounce
def crossed_counting_line_dir(prev_y, curr_y, line_y, min_px_after=LINE_DEBOUNCE_PX):
    if prev_y is None or curr_y is None or prev_y == curr_y:
        return 0
    prev_side = prev_y - line_y
    curr_side = curr_y - line_y
    crossed = (prev_side * curr_side) < 0 or (prev_side == 0 and curr_side != 0) or (curr_side == 0 and prev_side != 0)
    if not crossed:
        return 0
    if curr_y > line_y and (curr_y - line_y) >= min_px_after and curr_y > prev_y:
        return +1
    if curr_y < line_y and (line_y - curr_y) >= min_px_after and curr_y < prev_y:
        return -1
    return 0

# UI areas
BTN_POLY    = (200, 20, 300, 60)
BTN_START   = (320, 20, 420, 60)
BTN_LINE    = (440, 20, 540, 60)
BTN_EXIT    = (560, 20, 620, 60)

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
        cv2.putText(img, "Detection Area", (pts[0][0], max(pts[0][1] - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.line(img, (0, counting_line_y), (img.shape[1], counting_line_y), (0, 0, 255), 2)

def inside_button(x, y, rect):
    (x1, y1, x2, y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2

# =========================
# Global state & defaults
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
COUNT_DIR = 'down'
vehicle_count_by_class = defaultdict(int)
vehicle_count_by_group = defaultdict(int)
output_dir = "vehicle_captures"
counted_ids = set()
tracking_trails = defaultdict(list)
track_info = {}
_time_format = "%H:%M:%S"
_time_date_format = "%d/%m/%Y"
time_obj = datetime.strptime("08:00:00", _time_format).time()
end_time_obj = datetime.strptime("09:00:00", _time_format).time()
start_date = datetime.today().date()
start_time = datetime.combine(start_date, time_obj)
end_time = datetime.combine(start_date, end_time_obj)
capture_time = start_time
p1 = start_date.strftime(_time_date_format)
p2 = start_time.strftime(_time_format)
p3 = end_time.strftime(_time_format)
p4 = capture_time.strftime(_time_format)

size_thresholds = {
    'motorcycle': (5, 10),
    'car': (30, 50),
    'bus': (60, 100),
    'truck': (40, 90),
}

# =========================
# Snapshot restore utilities
# =========================
def restore_from_snapshot(attrs):
    global polygon_points, polygon_defined, poly_editing, polygon_mask
    global counting_line_y, line_mode, size_thresholds
    global start_date, p1, start_time, p2, end_time, p3, capture_time, p4
    sd = attrs.get("start_date")
    if sd:
        try:
            start_date = datetime.strptime(sd, _time_date_format).date() if "/" in sd else datetime.fromisoformat(sd).date()
            p1 = start_date.strftime(_time_date_format)
            print(f"[SNAPSHOT] start_date restored: {p1}")
        except Exception as e:
            print(f"[SNAPSHOT] start_date parse failed ({sd}): {e}")
    st = attrs.get("start_time")
    if st:
        try:
            fmt = _time_format if st.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(st, fmt).time()
            start_time = datetime.combine(start_date, t_val)
            p2 = start_time.strftime(_time_format)
            print(f"[SNAPSHOT] start_time restored: {start_time}")
        except Exception as e:
            print(f"[SNAPSHOT] start_time parse failed ({st}): {e}")
    et = attrs.get("end_time")
    if et:
        try:
            fmt = _time_format if et.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(et, fmt).time()
            end_time = datetime.combine(start_date, t_val)
            p3 = end_time.strftime(_time_format)
            print(f"[SNAPSHOT] end_time restored: {end_time}")
        except Exception as e:
            print(f"[SNAPSHOT] end_time parse failed ({et}): {e}")
    ct = attrs.get("capture_time")
    if ct:
        try:
            fmt = _time_format if ct.count(":") == 2 else "%H:%M"
            t_val = datetime.strptime(ct, fmt).time()
            capture_time = datetime.combine(start_date, t_val)
            p4 = capture_time.strftime(_time_format)
            print(f"[SNAPSHOT] capture_time restored: {capture_time}")
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
            w, h = int(v[0]), int(v[1])
            size_thresholds[k] = (w, h)
        except Exception:
            pass
    if line_mode == "AUTO" and polygon_defined:
        counting_line_y = polygon_mid_y(polygon_points)
    print("[SNAPSHOT] Restored:", f"ROI pts={len(polygon_points)} | line_mode={line_mode} y={counting_line_y} | thresholds={size_thresholds}")

def load_latest_snapshot(store_folder, filename_root=None):
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

# =========================
# Tk GUI bootstrap & I/O setup
# =========================
root = tk.Tk(); root.withdraw()
video_path = filedialog.askopenfilename(title="Select Video File or cancel to type URL", filetypes=[["Video files", "*.mp4 *.avi *.mov *.mkv"]])
if not video_path:
    video_path = simpledialog.askstring("Video URL", "Enter camera URL (e.g. rtsp://user:pass@host:554/stream):")
    if not video_path:
        print("[ERROR] No source provided. Exiting."); raise SystemExit

filename_only = make_safe_dir_name(video_path)
# Apply config again now that filename_only is known (for STREAM_URL substitution)
apply_config(cfg, filename_only_value=filename_only)

try:
    os.makedirs(output_dir, exist_ok=True)
except PermissionError:
    output_dir = os.path.join(os.path.expanduser("~"), "vehicle_captures"); os.makedirs(output_dir, exist_ok=True)
subfolder = os.path.join(output_dir, filename_only)
if os.path.isdir(subfolder):
    shutil.rmtree(subfolder)
os.makedirs(subfolder, exist_ok=True)
print(f"[IO] Saving crops, snapshots & logs under: {os.path.abspath(subfolder)}")

GPU_INFO = gpu_availability_report()

snap_path, snap_attrs = load_latest_snapshot(output_dir, filename_only)
if snap_attrs:
    print(f"[SNAPSHOT] Loading: {snap_path}")
    restore_from_snapshot(snap_attrs)
    start_processing = True
else:
    print("[SNAPSHOT] No snapshot JSON found; starting with defaults.")

# =========================
# YOLOv8
# =========================
DEVICE = select_device(); print(f"[DEVICE] Using {DEVICE}")
model = YOLO('yolov8s.pt')

centroid_tracker = CentroidTracker(max_distance=TRACK_MAX_DIST, max_disappeared=TRACK_MAX_DISAPPEAR)

# =========================
# Streaming configuration
# =========================
try:
    if (GPU_INFO.get("gpu_count", 0) > 0 or GPU_INFO.get("nvidia_smi")) and GPU_INFO.get("ffmpeg_nvenc"):
        if USE_NVENC:
            print("[GPU] NVENC available and enabled.")
except Exception:
    pass

cv2.namedWindow("Traffic Analysis")
current_mouse = (0, 0)
line_edit_mode = False

def mouse_callback(event, x, y, flags, param):
    global polygon_points, polygon_defined, poly_editing, current_mouse
    global running, start_processing, do_capture, line_edit_mode, counting_line_y, line_mode
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if inside_button(x, y, BTN_POLY):
            polygon_points = []; polygon_defined = False; poly_editing = True
            print("[INFO] Polygon edit: left-click add points, right-click or 'p' to close."); return
        if inside_button(x, y, BTN_START):
            start_processing = True; do_capture = True; ask_size_thresholds(); print("[INFO] Start clicked"); return
        if inside_button(x, y, BTN_LINE):
            line_edit_mode = True; print("[INFO] Line edit mode: click anywhere to set counting_line_y (press 'A' for AUTO). "); return
        if inside_button(x, y, BTN_EXIT):
            running = False; print("[INFO] Exit clicked"); return
        if poly_editing:
            polygon_points.append((x, y)); return
        if line_edit_mode:
            counting_line_y = y; line_mode = 'MANUAL'; line_edit_mode = False
            print(f"[INFO] Manual counting_line_y set to {counting_line_y}"); return
    if event == cv2.EVENT_RBUTTONDOWN:
        if poly_editing and len(polygon_points) >= 3:
            polygon_defined = True; poly_editing = False
            print(f"[INFO] Polygon defined with {len(polygon_points)} points")
            if 'frame_for_mask' in globals() and frame_for_mask is not None:
                rebuild_polygon_mask(frame_for_mask.shape)
            return

cv2.setMouseCallback("Traffic Analysis", mouse_callback)

if str(video_path).startswith("rtsp://"):
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;tcp|max_delay;0|reorder_queue_size;0|stimeout;3000000"

cap = cv2.VideoCapture(video_path)
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"[INFO] fps:{fps} total_frames: {total_frames}")

log_file_path = os.path.join(output_dir, f"{filename_only}_log.csv")
summary_file_path = os.path.join(output_dir, f"{filename_only}_summary_log.csv")
log_file = open(log_file_path, 'w', buffering=1); log_file.write("Timestamp,ID,Type,Confidence,VideoTime,Image\n")
summary_log_file = open(summary_file_path, 'w', buffering=1)
summary_log_file.write(f"{filename_only}\n"); summary_log_file.write(f"{p1}\n"); summary_log_file.write("Masa, Kelas 1, Kelas 2, Kelas 3, Kelas 4, Kelas 5, Kelas 6\n")

current_timestamp = start_time
check_timestamp = capture_time
previous_timestamp = capture_time
print(f"[INFO] current_timestamp: {current_timestamp} previous_timestamp: {previous_timestamp}")

last_crop = None; last_label = ""
last_good_frame = None; frame_for_mask = None

do_capture = False
streamer = None
stream_on = STREAM_ENABLED

read_failures = 0
reopen_backoff = 0.5
backoff_max = 8.0

t0_live = time.monotonic()
time_start = False; time_end = False

# Minute bucketing
def minute_bucket(dt: datetime, width_min: int = SUMMARY_BUCKET_MINUTES):
    minute = (dt.minute // width_min) * width_min
    return dt.replace(minute=minute, second=0, microsecond=0)

active_bucket = minute_bucket(current_timestamp, SUMMARY_BUCKET_MINUTES)
per_minute_counts = defaultdict(int)

# Ask sizes & times
def ask_size_thresholds():
    global p1,p2,p3,p4
    global start_date, start_time, end_time, capture_time
    global log_file, summary_log_file
    global current_timestamp, previous_timestamp
    global counted_ids, track_info, tracking_trails
    global active_bucket, per_minute_counts

    p1_local = simpledialog.askstring("Set Start Date", "Specify Date? (dd/mm/YYYY)", initialvalue=p1) or p1
    start_date_local = datetime.strptime(p1_local, _time_date_format).date()

    p2_local = simpledialog.askstring("Set Start Time", "Specify Start Time? (HH:MM:SS)", initialvalue=p2) or p2
    start_time_local = datetime.combine(start_date_local, datetime.strptime(p2_local, _time_format).time())

    p3_local = simpledialog.askstring("Set End Time", "Specify End Time? (HH:MM:SS)", initialvalue=p3) or p3
    end_time_local = datetime.combine(start_date_local, datetime.strptime(p3_local, _time_format).time())

    p4_local = simpledialog.askstring("Set capture Time", "Specify Capture Time? (HH:MM:SS)", initialvalue=p4) or p4
    capture_time_local = datetime.combine(start_date_local, datetime.strptime(p4_local, _time_format).time())

    start_date = start_date_local
    p1 = start_date.strftime(_time_date_format)
    start_time, end_time, capture_time = start_time_local, end_time_local, capture_time_local
    p2, p3, p4 = start_time.strftime(_time_format), end_time.strftime(_time_format), capture_time.strftime(_time_format)

    current_timestamp = start_time
    previous_timestamp = capture_time
    active_bucket = minute_bucket(current_timestamp, SUMMARY_BUCKET_MINUTES)

    for label in list(size_thresholds.keys()):
        w = simpledialog.askinteger("Set Min Width", f"Min width for {label}", initialvalue=size_thresholds[label][0], minvalue=0)
        h = simpledialog.askinteger("Set Min Height", f"Min height for {label}", initialvalue=size_thresholds[label][1], minvalue=0)
        size_thresholds[label] = (int(w or 0), int(h or 0))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    vehicle_count_by_class.clear(); vehicle_count_by_group.clear()
    counted_ids = set(); track_info.clear(); tracking_trails.clear(); per_minute_counts.clear()

    if os.path.isdir(subfolder): shutil.rmtree(subfolder)
    os.makedirs(subfolder, exist_ok=True)

    try: log_file.close()
    except Exception: pass
    try: summary_log_file.close()
    except Exception: pass

    log_path = os.path.join(output_dir, f"{filename_only}_log.csv")
    sum_path = os.path.join(output_dir, f"{filename_only}_summary_log.csv")
    globals()['log_file'] = open(log_path, 'w', buffering=1)
    globals()['summary_log_file'] = open(sum_path, 'w', buffering=1)
    log_file.write("Timestamp,ID,Type,Confidence,VideoTime,Image\n")
    summary_log_file.write(f"{filename_only}\n"); summary_log_file.write(f"{p1}\n"); summary_log_file.write("Masa, Kelas 1, Kelas 2, Kelas 3, Kelas 4, Kelas 5, Kelas 6\n")

while running:
    ret, frame = cap.read()
    if (not ret) or time_end:
        if total_frames == 0:
            read_failures += 1
            sleep_for = min(backoff_max, reopen_backoff * (2 ** min(read_failures, 4)))
            if read_failures % 25 == 0:
                print(f"[CAP] Reopening live source in {sleep_for:.1f}s...")
            time.sleep(sleep_for)
            cap.release(); cap = cv2.VideoCapture(video_path)
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            if stream_on and streamer and streamer.proc is None:
                streamer = None
            continue
        print("[INFO] COMPLETED"); break
    else:
        read_failures = 0

    frame = cv2.resize(frame, (640, 360))
    last_good_frame = frame.copy(); frame_for_mask = frame

    if poly_editing and len(polygon_points) >= 2:
        pass
    elif polygon_defined and polygon_mask is not None:
        pass
    elif polygon_defined:
        rebuild_polygon_mask(frame.shape)

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if total_frames > 0 else 0
    if total_frames > 0:
        current_time_sec = current_frame / fps; total_time_sec = total_frames / fps
    else:
        current_time_sec = time.monotonic() - t0_live; total_time_sec = 0.0
    current_time_str = time.strftime(_time_format, time.gmtime(current_time_sec))
    total_time_str = time.strftime(_time_format, time.gmtime(total_time_sec)) if total_time_sec else "--:--:--"

    if total_frames > 0 and current_frame == total_frames:
        do_capture = True

    time_obj_local = datetime.strptime(current_time_str, _time_format).time()
    delta = timedelta(hours=time_obj_local.hour, minutes=time_obj_local.minute, seconds=time_obj_local.second)
    result = start_time + delta
    if result > capture_time: time_start = True
    if result > end_time: time_end = True

    # bucket rollover
    next_bucket = minute_bucket(result, SUMMARY_BUCKET_MINUTES)
    if next_bucket != active_bucket:
        kelas1 = per_minute_counts['Class 1']; kelas2 = per_minute_counts['Class 2']
        kelas3 = per_minute_counts['Class 3']; kelas4 = per_minute_counts['Class 4']
        kelas5 = per_minute_counts['Class 5']; kelas6 = per_minute_counts['Class 6']
        summary_log_file.write(f"{active_bucket.strftime('%H:%M')} - {next_bucket.strftime('%H:%M')},{kelas1},{kelas2},{kelas3},{kelas4},{kelas5},{kelas6}\n")
        summary_log_file.flush(); per_minute_counts.clear(); active_bucket = next_bucket
        print(f"[INFO] {result}")  

    display = frame.copy(); draw_polygon_and_line(display)

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
        cv2.putText(fg_mask, f"{current_time_str} / {total_time_str}", (470, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

        detections = []
        tracks_this_frame = {}

        if USE_YOLO_TRACKER:
            with torch.inference_mode():
                res = model.track(frame, classes=[2,3,5,7], conf=YOLO_CONF, iou=YOLO_IOU, device=DEVICE, tracker='bytetrack.yaml', persist=True, verbose=False)[0]
            for box in res.boxes:
                if box.id is None: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0].item()); cls_id = int(box.cls[0].item()); obj_id = int(box.id[0].item())
                coco_name = model.names[cls_id]
                if coco_name in size_thresholds:
                    min_w, min_h = size_thresholds[coco_name]
                    if w < min_w or h < min_h: continue
                center = get_center(x1, y1, w, h)
                if not point_in_polygon(center): continue
                if cls_id == 2: label, color_rgb = "Class 1", (255, 255, 255)
                elif cls_id == 3: label, color_rgb = "Class 6", (255, 255, 0)
                elif cls_id == 5: label, color_rgb = "Class 5", (0, 0, 255)
                elif cls_id == 7: label, color_rgb = "Class 3", (255, 0, 0)
                else: label, color_rgb = "Class 2", (0, 0, 0)
                color_bgr = color_rgb[::-1]
                detections.append((x1, y1, x2, y2, label, conf, center, w, h, color_bgr, obj_id))
                tracks_this_frame[obj_id] = center
        else:
            with torch.inference_mode():
                res = model(frame, classes=[2,3,5,7], conf=YOLO_CONF, iou=YOLO_IOU, device=DEVICE, verbose=False)[0]
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0].item()); cls_id = int(box.cls[0].item())
                coco_name = model.names[cls_id]
                if coco_name in size_thresholds:
                    min_w, min_h = size_thresholds[coco_name]
                    if w < min_w or h < min_h: continue
                center = get_center(x1, y1, w, h)
                if not point_in_polygon(center): continue
                if cls_id == 2: label, color_rgb = "Class 1", (255, 255, 255)
                elif cls_id == 3: label, color_rgb = "Class 6", (255, 255, 0)
                elif cls_id == 5: label, color_rgb = "Class 5", (0, 0, 255)
                elif cls_id == 7: label, color_rgb = "Class 3", (255, 0, 0)
                else: label, color_rgb = "Class 2", (0, 0, 0)
                color_bgr = color_rgb[::-1]
                detections.append((x1, y1, x2, y2, label, conf, center, w, h, color_bgr, None))

            input_centroids = [det[6] for det in detections]
            objects = centroid_tracker.update(input_centroids)
            track_to_det = {}
            if detections:
                det_centers = [det[6] for det in detections]
                for objectID, trk_centroid in objects.items():
                    dists = [(i, (trk_centroid[0]-dcx)**2 + (trk_centroid[1]-dcy)**2) for i, (dcx, dcy) in enumerate(det_centers)]
                    if not dists: continue
                    best_i, best_d = min(dists, key=lambda t: t[1])
                    if np.sqrt(best_d) <= centroid_tracker.max_distance:
                        track_to_det[objectID] = best_i
                        tracks_this_frame[objectID] = det_centers[best_i]
            for objectID, det_idx in track_to_det.items():
                x1, y1, x2, y2, label, conf, center, w, h, color_bgr, _ = detections[det_idx]
                detections[det_idx] = (x1, y1, x2, y2, label, conf, center, w, h, color_bgr, objectID)

        alive_ids = set(tracks_this_frame.keys())
        for sid in list(track_info.keys()):
            if sid not in alive_ids:
                del track_info[sid]; tracking_trails.pop(sid, None)

        for det in detections:
            x1, y1, x2, y2, label, conf, center, w, h, color_bgr, obj_id = det
            if obj_id is None: continue
            if obj_id not in track_info:
                track_info[obj_id] = {"last_centroid": None, "counted": False, "last_y_bottom": None}

            prev_centroid = track_info[obj_id]["last_centroid"]
            already_counted = track_info[obj_id]["counted"]
            curr_y_bottom = y2
            prev_y_bottom = track_info[obj_id].get("last_y_bottom")

            if not already_counted:
                direction = crossed_counting_line_dir(prev_y=prev_y_bottom, curr_y=curr_y_bottom, line_y=counting_line_y, min_px_after=LINE_DEBOUNCE_PX)
                if ((COUNT_DIR == 'down' and direction == +1) or (COUNT_DIR == 'up' and direction == -1) or (COUNT_DIR == 'both' and direction != 0)):
                    track_info[obj_id]["counted"] = True
                    counted_ids.add(obj_id)
                    vehicle_count_by_class[label] += 1
                    vehicle_count_by_group[label] += 1
                    per_minute_counts[label] += 1
                    pad = 10 if label == "Class 6" else 5
                    x1p, y1p, x2p, y2p = max(0, x1-pad), max(0, y1-pad), min(frame.shape[1], x2+pad), min(frame.shape[0], y2+pad)
                    cropped = frame[y1p:y2p, x1p:x2p]
                    filename = os.path.join(subfolder, f"{label}_{obj_id}.jpg")
                    cv2.imwrite(filename, cropped)
                    log_file.write(f"{result.time()},{obj_id},{label},{conf:.2f},{current_time_str},{filename}\n")
                    log_file.flush()
                    last_crop = cv2.resize(cropped, (120, 70)) if cropped.size else None
                    last_label = f"{label} {w}x{h}"
                    print(f"filename: {filename} {result.time()} {last_label} {per_minute_counts[label]}")                    

            tracking_trails[obj_id].append(center)
            for pt in tracking_trails[obj_id][-TRAIL_LEN:]:
                cv2.circle(fg_mask, pt, 2, (160, 160, 160), -1)
            cv2.rectangle(display, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(display, f"{label}#{obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)

            track_info[obj_id]["last_centroid"] = center
            track_info[obj_id]["last_y_bottom"] = curr_y_bottom

        cv2.line(display, (0, counting_line_y), (640, counting_line_y), (0, 0, 255), 2)
        y_offset = 200
        for idx, (lbl, cnt) in enumerate(sorted(vehicle_count_by_class.items())):
            cv2.putText(display, f"{lbl}: {cnt}", (10, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if last_crop is not None:
            x_offset, y_offset = 520, 290
            h, w = last_crop.shape[:2]
            if y_offset + h <= display.shape[0] and x_offset + w <= display.shape[1]:
                display[y_offset:y_offset + h, x_offset:x_offset + w] = last_crop
                cv2.rectangle(display, (x_offset, y_offset - 20), (x_offset + 120, y_offset), (0, 0, 0), -1)
                cv2.putText(display, last_label, (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Foreground Mask", fg_mask)

    cv2.putText(display, f"'t'=Stream: {'ON' if stream_on else 'OFF'} -> {STREAM_URL if stream_on else ''}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0) if stream_on else (0,0,255), 1)
    cv2.putText(display, f"Dir keys: d=down, u=up, b=both  |  (u=undo while polygon editing) {COUNT_DIR.upper()}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(display, "Poly: L-click add, R-click p=close u=undo r=reset", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(display, "Line: l=line edit a=auto | General: c=capture, t=toggle stream", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if stream_on:
        if streamer is None or streamer.proc is None:
            streamer = FFmpegStreamer(STREAM_URL, fps=fps, width=display.shape[1], height=display.shape[0], use_nvenc=USE_NVENC)
            print(f"[STREAM] Publishing to {STREAM_URL}")
        streamer.write(display)

    cv2.imshow("Traffic Analysis", display)

    if 'do_capture' in globals() and do_capture:
        attrs = {
            "video_name": filename_only,
            "frame_index": current_frame,
            "total_frames": total_frames,
            "video_time": current_time_str,
            "total_time": total_time_str,
            "start_date": start_date.strftime(_time_date_format),
            "start_time": start_time.strftime(_time_format),
            "capture_time": capture_time.strftime(_time_format),
            "end_time": end_time.strftime(_time_format),
            "line_mode": line_mode,
            "counting_line_y": int(counting_line_y),
            "roi_points_count": len(polygon_points),
            "roi_points": [(int(x), int(y)) for x, y in polygon_points],
            "size_thresholds": {k: [int(v[0]), int(v[1])] for k, v in size_thresholds.items()}
        }
        save_snapshot(frame, attrs, output_dir, filename_only)
        do_capture = False

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
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
        polygon_points = []; polygon_defined = False; poly_editing = False; polygon_mask = None
        if line_mode == 'AUTO': counting_line_y = 250
    elif key == ord('p') and poly_editing and len(polygon_points) >= 3:
        polygon_defined = True; poly_editing = False; rebuild_polygon_mask(frame.shape)
    elif key == ord('l'):
        line_edit_mode = True; print("[INFO] Line edit mode: click anywhere to set counting_line_y (press 'A' for AUTO).")
    elif key == ord('a'):
        line_mode = 'AUTO'
        if polygon_defined: counting_line_y = polygon_mid_y(polygon_points)
        print(f"[INFO] Line mode set to AUTO (y={counting_line_y})")
    elif key == ord('c'):
        do_capture = True
    elif key == ord('t'):
        stream_on = not stream_on
        if not stream_on and streamer:
            streamer.close(); streamer = None
        print(f"[STREAM] {'ON' if stream_on else 'OFF'}")

# Cleanup
try: log_file.close()
except Exception: pass
try: summary_log_file.close()
except Exception: pass
if streamer: streamer.close()
cap.release(); cv2.destroyAllWindows()
