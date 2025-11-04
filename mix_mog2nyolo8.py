import cv2
import torch
import numpy as np
import time
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, simpledialog
from collections import defaultdict, OrderedDict
import math

# Centroid Tracker class
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

# Load YOLOv8 model (use ultralytics/yolov5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# Globals
running = True
start_processing = False
roi_defined = False
roi_start = (-1, -1)
roi_end = (-1, -1)
vehicle_count_by_class = defaultdict(int)
vehicle_count_by_group = defaultdict(int)
counting_line_y = 250
output_dir = "captured_objects"
os.makedirs(output_dir, exist_ok=True)

counted_ids = set()
tracking_trails = defaultdict(list)
tracker = CentroidTracker()
last_crop = None
last_label = ""

#current_timestamp = datetime.now().strftime("%H:%M")
#previous_timestamp = datetime.now().strftime("%H:%M")
# Create initial time
# Parse time
time_obj = datetime.strptime("08:00", "%H:%M").time()

# Combine with today's date
start_ptime = datetime.now()
#start_ptime = datetime.now().strftime("%H:%M")
start_time = datetime.combine(datetime.today(), time_obj)

#start_time = datetime.strptime("08:00", "%H:%M")
#current_timestamp = datetime.combine(datetime.today(), start_time.time())
#previous_timestamp = datetime.combine(datetime.today(), start_time.time())
current_timestamp = start_time
previous_timestamp = start_time
    
    
# Define size thresholds with default values
size_thresholds = {
    'motorcycle': (5, 10),
    'car': (30, 50),
    'bus': (60, 100),
    'truck': (40, 90),
}

# Tkinter dialog for thresholds
root = tk.Tk()
root.withdraw()

# File dialog to choose video
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[["Video files", "*.mp4 *.avi *.mov *.mkv"]])
if not video_path:
    print("[ERROR] No video selected. Exiting.")
    exit()
# Get only filename without extension
filename_only = os.path.splitext(os.path.basename(video_path))[0]


def ask_size_thresholds():
    for label in size_thresholds:
        w = simpledialog.askinteger("Set Width", f"Min width for {label}", initialvalue=size_thresholds[label][0], minvalue=0)
        h = simpledialog.askinteger("Set Height", f"Min height for {label}", initialvalue=size_thresholds[label][1], minvalue=0)
        size_thresholds[label] = (w or 0, h or 0)

    start_time_init = simpledialog.askstring("Set Start Time", "Specify Start Time?", initialvalue="08:00")
    # Parse time
    time_obj = datetime.strptime(start_time_init, "%H:%M").time()

    # Combine with today's date
    start_time = datetime.combine(datetime.today(), time_obj)

    #start_time = datetime.strptime(start_time_init, "%H:%M")

    current_timestamp = start_time
    previous_timestamp = start_time
    
    #current_timestamp = datetime.strptime(start_time_init, "%H:%M")
    #previous_timestamp = datetime.strptime(start_time_init, "%H:%M")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Reset all values to 0
    for key in vehicle_count_by_class:
        vehicle_count_by_class[key] = 0
        vehicle_count_by_group[key] = 0

    log_file = open(f"{filename_only}_log.csv", 'w')
    log_file.write("Timestamp,ID,Type,Confidence,Image\n")
    summary_log_file = open(f"{filename_only}_summary_log.csv", 'w')
    summary_log_file.write("Masa, Kelas 1, Kelas 2, Kelas 3, Kelas 4, Kelas 5, Kelas 6\n")

# GUI window
cv2.namedWindow("Traffic Analysis")

# Mouse callback
def mouse_callback(event, x, y, flags, param):
    global roi_defined, roi_start, roi_end, start_processing, running
    global top_left, bottom_right, start_click
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 420 and 20 <= y <= 60:
            #ask_size_thresholds()
            start_processing = True
            ask_size_thresholds()
            print("[INFO] Start clicked")
        elif 140 <= x <= 540 and 20 <= y <= 60:
            running = False
            print("[INFO] Exit clicked")
        else:
            start_click = True
            roi_start = (x, y)
            roi_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags == cv2.EVENT_FLAG_LBUTTON:
            roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if start_click:
            roi_end = (x, y)
            roi_defined = True        
            print(f"[INFO] ROI defined from {roi_start} to {roi_end}")
            for key in vehicle_count_by_class:
                print(f"[INFO] {key}")
            start_click = False
            

cv2.setMouseCallback("Traffic Analysis", mouse_callback)

# Helper
def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

# Video and background subtractor
cap = cv2.VideoCapture(video_path)
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# CSV logging
log_file = open(f"{filename_only}_log.csv", 'w')
log_file.write("Timestamp,ID,Type,Confidence,fps,Image\n")
summary_log_file = open(f"{filename_only}_summary_log.csv", 'w')
summary_log_file.write("Masa, Kelas 1, Kelas 2, Kelas 3, Kelas 4, Kelas 5, Kelas 6\n")

# Main loop
while running:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] COMPLETED")
        break
    frame = cv2.resize(frame, (640, 360))

    # Current frame position
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    #start_time += timedelta(minutes=1)
    #print(f"[INFO] {start_time}")
    # Get current time
    now = datetime.now()

    # Calculate difference in minutes
    diff_minutes = (now - start_ptime).total_seconds() / 60

    if diff_minutes > 1:
        print(f"Previous time: {start_ptime}")
        print(f"Now:           {now}")
        print(f"Difference:    {diff_minutes:.2f} minutes")
    
        start_ptime = now
    # Time calculations
    current_time_sec = current_frame / fps
    total_time_sec = total_frames / fps
    current_time_str = time.strftime("%M:%S", time.gmtime(current_time_sec))
    total_time_str = time.strftime("%M:%S", time.gmtime(total_time_sec))

    
    # Draw GUI buttons
    cv2.rectangle(frame, (320, 20), (420, 60), (0, 200, 0), -1)
    cv2.putText(frame, "Start", (340, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(frame, (440, 20), (540, 60), (0, 0, 200), -1)
    cv2.putText(frame, "Exit", (465, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw ROI
    if roi_defined:
        top_left = (min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1]))
        bottom_right = (max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
        cv2.putText(frame, "Detection Area", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        counting_line_y = (top_left[1] + bottom_right[1]) // 2
        #print(f"[INFO] Define detection area")

    if start_processing and roi_defined:
        fg_mask = back_sub.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw progress on screen
        cv2.putText(fg_mask,
                f"{current_time_str} / {total_time_str}",
                (500, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

        motion_boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 400:
                x, y, w, h = cv2.boundingRect(cnt)
                center = get_center(x, y, w, h)
                if roi_start[0] <= center[0] <= roi_end[0] and roi_start[1] <= center[1] <= roi_end[1]:
                    motion_boxes.append((x, y, w, h))

        results = model(frame)
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            # Filter by width and height thresholds
            if label in size_thresholds:
                min_w, min_h = size_thresholds[label]
                if w < min_w or h < min_h:
                    continue
                
            if int(cls) == 2:
                label = f"Class 1"
                color_rgb = (255, 255, 255)  # RGB white
            elif int(cls) == 3:
                label = f"Class 6"
                color_rgb = (255, 255, 0)  # RGB yellow
            elif int(cls) == 5:
                label = f"Class 5"
                color_rgb = (0, 0, 255)  # RGB blue
            elif int(cls) == 7:
                label = f"Class 3"
                color_rgb = (255, 0, 0)  # RGB red
            else:
                label = f"Class 2"
                color_rgb = (0, 0, 0)  # RGB black
            color_bgr = color_rgb[::-1]  # Reverse to BGR for OpenCV
            
            x1, y1, x2, y2 = map(int, xyxy)
            w, h = x2 - x1, y2 - y1
            center = get_center(x1, y1, w, h)
            if roi_defined and not (roi_start[0] <= center[0] <= roi_end[0] and roi_start[1] <= center[1] <= roi_end[1]):
                continue            
                
            detections.append((x1, y1, x2, y2, label, conf, center, w, h))

        input_centroids = [det[6] for det in detections]
        objects = tracker.update(input_centroids)

        for (objectID, centroid), det in zip(objects.items(), detections):
            x1, y1, x2, y2, label, conf, center, w, h = det

            if objectID not in counted_ids and y1 < counting_line_y < y2:
                counted_ids.add(objectID)
                vehicle_count_by_class[label] += 1

                #timestamp = datetime.now().strftime("%H:%M")
                current_timestamp = start_time
                print(f"[INFO] {current_timestamp}")
                #current_timestamp = datetime.combine(datetime.today(), start_time.time())
                if not current_timestamp == previous_timestamp:
                    #car
                    kelas1 = vehicle_count_by_group["Class 1"]
                    kelas2 = 0
                    #van/lorry
                    kelas3 = vehicle_count_by_group["Class 3"]
                    kelas4 = 0
                    #bus
                    kelas5 = vehicle_count_by_group["Class 5"]
                    #motorcycle
                    kelas6 = vehicle_count_by_group["Class 6"]
                    summary_log_file.write(f"{previous_timestamp.time()} - {current_timestamp.time()},")
                    summary_log_file.write(f"{kelas1},{kelas2},{kelas3},{kelas4},{kelas5},{kelas6}\n")
                    # Force flush to disk (optional if buffering=1)
                    summary_log_file.flush()
                    # Reset all values to 0
                    for key in vehicle_count_by_class:
                        vehicle_count_by_group[key] = 0
                    previous_timestamp = current_timestamp
                else:
                    vehicle_count_by_group[label] += 1
                cropped = frame[y1:y2, x1:x2]
                filename = f"{output_dir}/{label}_{vehicle_count_by_class[label]}.jpg"
                cv2.imwrite(filename, cropped)
                log_file.write(f"{current_timestamp.time()},{objectID},{label},{float(conf):.2f},{current_time_str},{filename}\n")
                log_file.flush

                # Save last cropped image for display
                last_crop = cv2.resize(cropped, (120, 70))
                last_label = f"{label} {w}X{h}"

            # Add centroid to trail and draw dots
            tracking_trails[objectID].append(centroid)
            for pt in tracking_trails[objectID]:
                cv2.circle(frame, pt, 2, (160, 160, 160), -1)  # gray dot

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 2)
            #cv2.circle(frame, centroid, 4, (255, 0, 0), -1)      
        
        # Counting line and stats
        cv2.line(frame, (0, counting_line_y), (640, counting_line_y), (0, 0, 255), 2)
        y_offset = 200
        
        for idx, (label, count) in enumerate(vehicle_count_by_class.items()):
            #min_w, min_h = size_thresholds[label]
            cv2.putText(frame, f"{label}: {count}", (10, y_offset + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show last cropped image preview
        if last_crop is not None:
            x_offset, y_offset = 520, 290
            frame[y_offset:y_offset + last_crop.shape[0], x_offset:x_offset + last_crop.shape[1]] = last_crop
            cv2.rectangle(frame, (x_offset, y_offset - 20), (x_offset + 100, y_offset), (0, 0, 0), -1)
            cv2.putText(frame, last_label, (x_offset, y_offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Foreground Mask", fg_mask)

    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

log_file.close()
summary_log_file.close()
cap.release()
cv2.destroyAllWindows()
