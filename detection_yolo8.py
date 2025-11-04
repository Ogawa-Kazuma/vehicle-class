import cv2
import torch
import numpy as np
import time
import os
import json
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, simpledialog
from collections import defaultdict, OrderedDict
import math
from ultralytics import YOLO

# ---------------------------
# Centroid Tracker class
# ---------------------------
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
        del self.objects[objectID]
        del self.disappeared[objectID]

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

# ---------------------------
# Load YOLOv8 model (COCO)
# ---------------------------
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8s.pt')  # swap to 'yolov8s.pt' for more accuracy if GPU allows

# ---------------------------
# Globals
# ---------------------------
running = True
start_processing = False

# Polygon ROI state
polygon_points = []          # list of (x,y)
polygon_defined = False
poly_editing = False
polygon_mask = None
current_mouse = (0, 0)

# Counting line state
counting_line_y = 250
line_mode = 'AUTO'          # 'AUTO' (polygon midpoint) or 'MANUAL'
line_edit_mode = False      # when True, next click sets counting_line_y

vehicle_count_by_class = defaultdict(int)
vehicle_count_by_group = defaultdict(int)
output_dir = "vehicle_captures"
os.makedirs(output_dir, exist_ok=True)

counted_ids = set()
tracking_trails = defaultdict(list)
tracker = CentroidTracker()
last_crop = None
last_label = ""
do_capture = False          # set True to save snapshot this frame

# Time setup
time_obj = datetime.strptime("08:00", "%H:%M").time()
start_date = datetime.today().date()
p1 = datetime.combine(start_date, time_obj).strftime("%d/%m/%Y")
start_time = datetime.combine(start_date, time_obj)
p2 = start_time.strftime("%H:%M")

# Define size thresholds with default values (min width, min height)
size_thresholds = {
    'motorcycle': (5, 10),
    'car': (30, 50),
    'bus': (100, 100),
    'truck': (50, 90),
}

# ---------------------------
# Helpers / Utilities
# ---------------------------
def polygon_mid_y(pts):
    ys = [p[1] for p in pts]
    return (min(ys) + max(ys)) // 2 if ys else 250

def rebuild_polygon_mask(frame_shape):
    """Builds/updates the binary mask for the polygon ROI and refreshes counting_line_y if AUTO."""
    global polygon_mask, counting_line_y
    polygon_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if len(polygon_points) >= 3:
        pts = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(polygon_mask, [pts], 255)
        if line_mode == 'AUTO':
            counting_line_y = polygon_mid_y(polygon_points)

def get_last_valid_frame_index(video_path: str):
    cap_local = cv2.VideoCapture(video_path)
    if not cap_local.isOpened():
        raise RuntimeError("Cannot open video")

    n = int(cap_local.get(cv2.CAP_PROP_FRAME_COUNT))
    if n > 0:
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok, _ = cap_local.read()
        if ok:
            cap_local.release()
            return n - 1, n

    lo, hi = 0, 1
    while True:
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, hi)
        ok, _ = cap_local.read()
        if not ok:
            break
        lo, hi = hi, hi * 2
        if hi > 1_000_000_000:
            break

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        cap_local.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, _ = cap_local.read()
        if ok:
            lo = mid
        else:
            hi = mid

    cap_local.set(cv2.CAP_PROP_POS_FRAMES, lo)
    ok, _ = cap_local.read()
    cap_local.release()
    if not ok:
        raise RuntimeError("Could not locate a readable last frame")
    return lo, lo + 1

def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def point_in_polygon(pt):
    if not polygon_defined or len(polygon_points) < 3:
        return True  # if no polygon, accept all
    pts = np.array(polygon_points, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(pt[0]), float(pt[1])), False) >= 0

def draw_buttons(img):
    #draw_btn(img, BTN_CAPTURE, "Capture", (60, 60, 60))
    draw_btn(img, BTN_POLY, "Poly", (100, 100, 0))
    draw_btn(img, BTN_START, "Start", (0, 200, 0))
    draw_btn(img, BTN_LINE, "Line", (200, 100, 0))
    draw_btn(img, BTN_EXIT, "Exit", (0, 0, 200))

def draw_btn(img, rect, label, color):
    (x1, y1, x2, y2) = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.putText(img, label, (x1 + 8, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_polygon_and_line(img):
    if len(polygon_points) > 0:
        # vertices + edges
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

    # counting line (full width; not clipped)
    cv2.line(img, (0, counting_line_y), (img.shape[1], counting_line_y), (0, 0, 255), 2)

def draw_info_panel(img, info_lines, origin=(10, 10)):
    # background box sized to text
    wmax = 0
    for line in info_lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        wmax = max(wmax, tw)
    pad = 6
    line_h = 18
    box_w = wmax + pad * 2
    box_h = line_h * len(info_lines) + pad * 2
    x, y = origin
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (255, 255, 255), 1)
    for i, line in enumerate(info_lines):
        ly = y + pad + (i + 1) * line_h - 4
        cv2.putText(img, line, (x + pad, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def save_snapshot(base_frame, attrs, save_dir, filename_root):
    """Save a snapshot image WITHOUT UI buttons, but WITH polygon fill, counting line, and info panel.
       Also writes a JSON sidecar with the same attributes."""
    snap = base_frame.copy()

    # Draw polygon & line (no buttons)
    draw_polygon_and_line(snap)

    # Info panel
    info_lines = [
        f"Date: {attrs['start_date']}",
        f"Video: {attrs['video_time']} / {attrs['total_time']} (f{attrs['frame_index']}/{attrs['total_frames']})",
        f"Line: {attrs['line_mode']} y={attrs['counting_line_y']}",
        f"ROI pts: {attrs['roi_points_count']}",
        f"Min sizes: mc={size_thresholds.get('motorcycle', (0,0))}, car={size_thresholds.get('car',(0,0))}",
        f"           bus={size_thresholds.get('bus',(0,0))}, truck={size_thresholds.get('truck',(0,0))}"
    ]
    #draw_info_panel(snap, info_lines, origin=(10, 10))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(save_dir, f"{filename_root}.jpg")
    json_path = img_path.replace(".jpg", ".json")

    cv2.imwrite(img_path, snap)
    with open(json_path, 'w') as jf:
        json.dump(attrs, jf, indent=2)

    print(f"[SNAPSHOT] Saved: {img_path}")
    print(f"[SNAPSHOT] Meta  : {json_path}")

# ---------------------------
# Buttons & mouse
# ---------------------------
def inside_button(x, y, rect):
    (x1, y1, x2, y2) = rect
    return x1 <= x <= x2 and y1 <= y <= y2

BTN_CAPTURE = (80, 20, 180, 60)
BTN_POLY    = (200, 20, 300, 60)
BTN_START   = (320, 20, 420, 60)
BTN_LINE    = (440, 20, 540, 60)
BTN_EXIT    = (560, 20, 620, 60)

def ask_size_thresholds():
    global start_date, start_time
    global log_file, summary_log_file
    global current_timestamp, previous_timestamp

    start_date_init = simpledialog.askstring("Set Start Date", "Specify Date? (dd/mm/YYYY)", initialvalue=p1)
    start_date = datetime.strptime(start_date_init, "%d/%m/%Y").date()
    #print(f"[INFO] {start_date}")

    start_time_init = simpledialog.askstring("Set Start Time", "Specify Time? (HH:MM)", initialvalue=p2)
    time_obj_local = datetime.strptime(start_time_init, "%H:%M").time()
    start_time = datetime.combine(start_date, time_obj_local)
    #print(f"[INFO] {start_time}")

    current_timestamp = start_time
    previous_timestamp = start_time

    for label in size_thresholds:
        w = simpledialog.askinteger("Set Min Width", f"Min width for {label}", initialvalue=size_thresholds[label][0], minvalue=0)
        h = simpledialog.askinteger("Set Min Height", f"Min height for {label}", initialvalue=size_thresholds[label][1], minvalue=0)
        size_thresholds[label] = (w or 0, h or 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Reset counts
    for key in list(vehicle_count_by_class.keys()):
        vehicle_count_by_class[key] = 0
    for key in list(vehicle_count_by_group.keys()):
        vehicle_count_by_group[key] = 0

    # Reopen logs (overwrite) with headers
    try:
        log_file.close()
    except:
        pass
    try:
        summary_log_file.close()
    except:
        pass

    log_file_path = f"{filename_only}_log.csv"
    summary_file_path = f"{filename_only}_summary_log.csv"

    log_file = open(log_file_path, 'w', buffering=1)
    log_file.write("Timestamp,ID,Type,Confidence,VideoTime,Image\n")

    summary_log_file = open(summary_file_path, 'w', buffering=1)
    summary_log_file.write(f"{filename_only}\n")
    summary_log_file.write(f"{start_date_init}\n")
    summary_log_file.write("Masa, Kelas 1, Kelas 2, Kelas 3, Kelas 4, Kelas 5, Kelas 6\n")

def mouse_callback(event, x, y, flags, param):
    global polygon_points, polygon_defined, poly_editing, current_mouse
    global running, start_processing, line_edit_mode, counting_line_y, line_mode, do_capture

    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Buttons
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

        # Actions on frame
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
        
LINE_BAND_PX = 4  # tolerance band around counting_line_y to catch near-miss frames
track_info = {}   # objectID -> {"last_centroid": (x,y) or None, "counted": False}

def crossed_counting_line(prev_y, curr_y, line_y, band=0):
    """Return True if the segment prev_y->curr_y crosses y=line_y (with optional band)."""
    if prev_y is None or curr_y is None:
        return False
    # Expand line into a band [line_y-band, line_y+band]
    low, high = line_y - band, line_y + band
    # If previous and current are on opposite sides (strict crossing)
    if (prev_y - line_y) * (curr_y - line_y) < 0:
        return True
    # If either end lies inside the band and the other end is on the other side or also in band
    if (low <= prev_y <= high) or (low <= curr_y <= high):
        # Consider it a crossing if there's motion across the band
        return prev_y < low and curr_y > high or prev_y > high and curr_y < low or (low <= prev_y <= high and prev_y != curr_y)
    return False

# ---------------------------
# Tkinter dialog
# ---------------------------
root = tk.Tk()
root.withdraw()

# ---------------------------
# File dialog to choose video
# ---------------------------
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[["Video files", "*.mp4 *.avi *.mov *.mkv"]])
if not video_path:
    print("[ERROR] No video selected. Exiting.")
    raise SystemExit

filename_only = os.path.splitext(os.path.basename(video_path))[0]
subfolder = f"{output_dir}/{filename_only}"
os.makedirs(subfolder, exist_ok=True)

# ---------------------------
# GUI window
# ---------------------------
cv2.namedWindow("Traffic Analysis")
cv2.setMouseCallback("Traffic Analysis", mouse_callback)

# ---------------------------
# Video & background subtractor
# ---------------------------
cap = cv2.VideoCapture(video_path)
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # guard fallback
print(f"[INFO] fps:{fps} total_frames: {total_frames}")

# ---------------------------
# CSV logging (init with headers)
# ---------------------------
log_file = open(f"{filename_only}_log.csv", 'w', buffering=1)
log_file.write("Timestamp,ID,Type,Confidence,VideoTime,Image\n")
summary_log_file = open(f"{filename_only}_summary_log.csv", 'w', buffering=1)

# ---------------------------
# Time trackers
# ---------------------------
current_timestamp = start_time
previous_timestamp = start_time

# ---------------------------
# Main loop
# ---------------------------
#last_good_frame = None
frame_for_mask = None

while running:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] COMPLETED")
        break

    frame = cv2.resize(frame, (640, 360))
    #last_good_frame = frame.copy()
    frame_for_mask = frame  # for building mask with correct shape

    if poly_editing and len(polygon_points) >= 2:
        pass
    elif polygon_defined and polygon_mask is None:
        rebuild_polygon_mask(frame.shape)

    # Current frame position
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # Time calculations
    current_time_sec = current_frame / fps
    total_time_sec = (total_frames / fps) if total_frames > 0 else 0
    current_time_str = time.strftime("%H:%M:%S", time.gmtime(current_time_sec))
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time_sec)) if total_time_sec else "00:00:00"

    if current_frame == total_frames:
        do_capture = True
        
    # Advance wall-clock label based on video time
    ctime = time.strftime("%H:%M:%S", time.gmtime(current_time_sec))
    time_obj_local = datetime.strptime(ctime, "%H:%M:%S").time()
    delta = timedelta(hours=time_obj_local.hour, minutes=time_obj_local.minute, seconds=time_obj_local.second)
    result = start_time + delta

    if int(time_obj_local.second) == 0 and int(time_obj_local.minute) > 0:
        if current_timestamp != result:
            current_timestamp = result
            print(f"[INFO] {current_timestamp}")

    # ---- DISPLAY FRAME (with UI buttons) ----
    display = frame.copy()
    # polygon/line preview (so you can see while editing)
    draw_polygon_and_line(display)
    draw_buttons(display)

    if start_processing and polygon_defined:
        # Motion mask (masked to polygon)
        fg_mask = back_sub.apply(frame)
        if polygon_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, polygon_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw progress on mask
        cv2.putText(fg_mask, f"{current_time_str} / {total_time_str}", (470, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

        motion_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:
                x, y, w, h = cv2.boundingRect(cnt)
                center = get_center(x, y, w, h)
                if point_in_polygon(center):
                    motion_boxes.append((x, y, w, h))

        # ---------------------------
        # YOLOv8 inference (car=2, motorcycle=3, bus=5, truck=7)
        # ---------------------------
        res = model(frame, classes=[2, 3, 5, 7], conf=0.25, iou=0.45, device=device, verbose=False)[0]

        detections = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            coco_name = model.names[cls_id]  # 'car','motorcycle','bus','truck'

            # Size filter (after we have w/h)
            if coco_name in size_thresholds:
                min_w, min_h = size_thresholds[coco_name]
                if w < min_w or h < min_h:
                    continue

            center = get_center(x1, y1, w, h)
            if not point_in_polygon(center):
                continue

            # Map to your class buckets & colors
            if cls_id == 2:      # car
                label, color_rgb = "Class 1", (255, 255, 255)   # white
            elif cls_id == 3:    # motorcycle
                label, color_rgb = "Class 6", (255, 255, 0)     # yellow
            elif cls_id == 5:    # bus
                label, color_rgb = "Class 5", (0, 0, 255)       # blue
            elif cls_id == 7:    # truck
                label, color_rgb = "Class 3", (255, 0, 0)       # red
            else:
                label, color_rgb = "Class 2", (0, 0, 0)         # black

            color_bgr = color_rgb[::-1]
            detections.append((x1, y1, x2, y2, label, conf, center, w, h, color_bgr))

        # Tracking
        input_centroids = [det[6] for det in detections]
        objects = tracker.update(input_centroids)

        # ---------------------------
        # Tracking association: map each track to its closest detection by center (fixes mismatch)
        # ---------------------------
        track_to_det = {}  # objectID -> det_index
        if detections:
            det_centers = [det[6] for det in detections]  # list of (cx, cy)
            unused = set(range(len(detections)))
            for objectID, centroid in objects.items():
                # pick nearest unused detection to this track centroid
                best_i, best_d = None, float("inf")
                for i in list(unused):
                    dcx, dcy = det_centers[i]
                    d = (centroid[0] - dcx)**2 + (centroid[1] - dcy)**2  # squared dist
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None:
                    track_to_det[objectID] = best_i
                    unused.discard(best_i)

        for (objectID, centroid), det in zip(objects.items(), detections):
            x1, y1, x2, y2, label, conf, center, w, h, color_bgr = det

            # Count when crossing the horizontal line
            if objectID not in counted_ids and y1 < counting_line_y < y2:
                counted_ids.add(objectID)
                vehicle_count_by_class[label] += 1

                if current_timestamp != previous_timestamp:
                    # Build grouped summary rows
                    kelas1 = vehicle_count_by_group["Class 1"]
                    kelas2 = 0
                    kelas3 = vehicle_count_by_group["Class 3"]
                    kelas4 = 0
                    kelas5 = vehicle_count_by_group["Class 5"]
                    kelas6 = vehicle_count_by_group["Class 6"]
                    summary_log_file.write(f"{previous_timestamp.time()} - {current_timestamp.time()},"
                                           f"{kelas1},{kelas2},{kelas3},{kelas4},{kelas5},{kelas6}\n")
                    summary_log_file.flush()
                    for key in list(vehicle_count_by_group.keys()):
                        vehicle_count_by_group[key] = 0
                    previous_timestamp = current_timestamp
                else:
                    vehicle_count_by_group[label] += 1

                cropped = frame[y1:y2, x1:x2]
                filename = f"{output_dir}/{filename_only}/{label}_{vehicle_count_by_class[label]}.jpg"
                cv2.imwrite(filename, cropped)
                log_file.write(f"{current_timestamp.time()},{objectID},{label},{conf:.2f},{current_time_str},{filename}\n")
                log_file.flush()

                # Preview
                last_crop = cv2.resize(cropped, (120, 70))
                last_label = f"{label} {w}x{h}"

            # Draw trail
            tracking_trails[objectID].append(centroid)
            for pt in tracking_trails[objectID]:
                cv2.circle(display, pt, 2, (160, 160, 160), -1)  # gray dot

            # Draw bbox & label on display (not on base frame)
            cv2.rectangle(display, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(display, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 2)

        # Counting line & stats on display
        cv2.line(display, (0, counting_line_y), (640, counting_line_y), (0, 0, 255), 2)
        y_offset = 200
        for idx, (label, count) in enumerate(vehicle_count_by_class.items()):
            cv2.putText(display, f"{label}: {count}", (10, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show last cropped image preview (on display only)
        if last_crop is not None:
            x_offset, y_offset = 520, 290
            display[y_offset:y_offset + last_crop.shape[0], x_offset:x_offset + last_crop.shape[1]] = last_crop
            cv2.rectangle(display, (x_offset, y_offset - 20), (x_offset + 120, y_offset), (0, 0, 0), -1)
            cv2.putText(display, last_label, (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Foreground Mask", fg_mask)

    # Short instructions on display
    cv2.putText(display, "Poly: L-click add, R-click/'p' close | 'u'=undo 'r'=reset | 'l'=line edit, 'a'=auto, 'c'=capture",
                (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Show main window
    cv2.imshow("Traffic Analysis", display)

    # Handle snapshot request (no buttons in saved image)
    if do_capture:
        attrs = {
            "video_name": filename_only,
            "frame_index": current_frame,
            "total_frames": total_frames,
            "video_time": current_time_str,
            "total_time": total_time_str,
            "start_date": start_date.strftime("%d/%m/%Y"),
            "line_mode": line_mode,
            "counting_line_y": int(counting_line_y),
            "roi_points_count": len(polygon_points),
            "roi_points": [(int(x), int(y)) for x, y in polygon_points],
            "size_thresholds": {k: [int(v[0]), int(v[1])] for k, v in size_thresholds.items()}
        }
        # use base frame (no buttons), then draw polygon + line + info panel
        save_snapshot(frame, attrs, subfolder, filename_only)
        do_capture = False

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('u') and poly_editing and polygon_points:
        polygon_points.pop()
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

# Cleanup
try:
    log_file.close()
except:
    pass
try:
    summary_log_file.close()
except:
    pass
cap.release()
cv2.destroyAllWindows()
