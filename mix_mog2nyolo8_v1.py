import cv2
import torch
import numpy as np
import time
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict, OrderedDict
import threading

# Centroid Tracker
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.trails = defaultdict(list)
        self.max_distance = max_distance

    def update(self, input_centroids):
        updated_objects = OrderedDict()

        if len(self.objects) == 0:
            for centroid in input_centroids:
                updated_objects[self.nextObjectID] = centroid
                self.trails[self.nextObjectID].append(centroid)
                self.nextObjectID += 1
        else:
            for objectID, centroid in self.objects.items():
                min_dist = float('inf')
                match_centroid = None
                for c in input_centroids:
                    dist = np.linalg.norm(np.array(centroid) - np.array(c))
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        match_centroid = c
                if match_centroid:
                    updated_objects[objectID] = match_centroid
                    self.trails[objectID].append(match_centroid)
                    input_centroids.remove(match_centroid)
            for c in input_centroids:
                updated_objects[self.nextObjectID] = c
                self.trails[self.nextObjectID].append(c)
                self.nextObjectID += 1

        self.objects = updated_objects
        return self.objects

# Default size thresholds
size_thresholds = {
    'motorcycle': [30, 30],
    'car': [60, 60],
    'bus': [100, 100],
    'truck': [100, 100],
}

# Threshold GUI (always shown and updated during processing)
def update_thresholds_live():
    threshold_window = tk.Toplevel()
    threshold_window.title("Adjust Size Thresholds")
    entries = {}

    def update_values():
        for lbl in size_thresholds:
            try:
                w = int(entries[lbl][0].get())
                h = int(entries[lbl][1].get())
                size_thresholds[lbl] = [w, h]
            except ValueError:
                continue
        threshold_window.after(1000, update_values)

    row = 0
    for label in size_thresholds:
        tk.Label(threshold_window, text=label.title()).grid(row=row, column=0)
        w_entry = tk.Entry(threshold_window, width=5)
        w_entry.insert(0, str(size_thresholds[label][0]))
        w_entry.grid(row=row, column=1)
        h_entry = tk.Entry(threshold_window, width=5)
        h_entry.insert(0, str(size_thresholds[label][1]))
        h_entry.grid(row=row, column=2)
        entries[label] = (w_entry, h_entry)
        row += 1

    update_values()  # Start periodic updates
    threshold_window.mainloop()

# Launch the GUI for thresholds in a separate thread (always running)
thresh_thread = threading.Thread(target=update_thresholds_live, daemon=True)
thresh_thread.start()

# Load YOLOv8 (from Ultralytics YOLOv5 repo)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [2, 3, 5, 7]  # Car, motorcycle, bus, truck

# File Dialog to select video
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
if not video_path:
    print("[ERROR] No video selected. Exiting.")
    exit()

cap = cv2.VideoCapture(video_path)
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
tracker = CentroidTracker()

vehicle_count = defaultdict(int)
counted_ids = set()

counting_line_y = 180
offset = 15

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 360))

    fg_mask = back_sub.apply(frame)
    results = model(frame)
    detections = results.pandas().xyxy[0]

    centers = []
    for _, row in detections.iterrows():
        cls = row['name']
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        w, h = x2 - x1, y2 - y1

        if cls in size_thresholds:
            min_w, min_h = size_thresholds[cls]
            if w >= min_w and h >= min_h:
                cx, cy = x1 + w // 2, y1 + h // 2
                centers.append((cx, cy))
                color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    objects = tracker.update(centers)

    for objectID, center in objects.items():
        cx, cy = center
        if objectID not in counted_ids and abs(cy - counting_line_y) < offset:
            counted_ids.add(objectID)
            label = None
            for _, row in detections.iterrows():
                rcx = int((row['xmin'] + row['xmax']) / 2)
                rcy = int((row['ymin'] + row['ymax']) / 2)
                if abs(cx - rcx) < 20 and abs(cy - rcy) < 20:
                    label = row['name']
                    break
            if label:
                vehicle_count[label] += 1

        cv2.putText(frame, f'ID {objectID}', (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Draw trail
        for point in tracker.trails[objectID][-30:]:
            cv2.circle(frame, point, 2, (160, 160, 160), -1)

    # Draw counting line
    cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (255, 0, 255), 2)

    # Display counts
    y_offset = 20
    for cls, count in vehicle_count.items():
        cv2.putText(frame, f'{cls}: {count}', (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
