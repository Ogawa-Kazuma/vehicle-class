import cv2
import time
import numpy as np
from collections import OrderedDict
import math
import csv
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog

class CentroidTracker:
    def __init__(self, max_distance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.max_distance = max_distance

    def update(self, input_centroids):
        updated_objects = OrderedDict()

        if len(self.objects) == 0:
            for centroid in input_centroids:
                updated_objects[self.nextObjectID] = centroid
                self.nextObjectID += 1
        else:
            for objectID, centroid in self.objects.items():
                min_dist = float('inf')
                match_centroid = None
                for c in input_centroids:
                    dist = math.hypot(centroid[0] - c[0], centroid[1] - c[1])
                    if dist < min_dist and dist < self.max_distance:
                        min_dist = dist
                        match_centroid = c
                if match_centroid:
                    updated_objects[objectID] = match_centroid
                    input_centroids.remove(match_centroid)

            for c in input_centroids:
                updated_objects[self.nextObjectID] = c
                self.nextObjectID += 1

        self.objects = updated_objects
        return self.objects

# Define detection boundary (ROI) - adjust as needed
ROI_TOP_LEFT = (150, 200)
ROI_BOTTOM_RIGHT = (500, 300)

# Global flags
running = True
start_processing = False

drawing_roi = False
roi_defined = False
roi_start = (-1, -1)
roi_end = (-1, -1)

# Video path
#video_path = 'output.mp4'

# Output directory
output_dir = "captured_objects"
os.makedirs(output_dir, exist_ok=True)


# GUI file dialog to select video
root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    print("[ERROR] No video selected. Exiting.")
    exit()
print(f"[INFO] Selected video: {video_path}")

cap = cv2.VideoCapture(video_path)
#cap = cv2.VideoCapture(video_path)

# Background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Constants
#SMALL_AREA = 550
#MEDIUM_AREA = 4000
frame_count = 0
total_count = 0
last_count = 0
start_time = time.time()
counting_line_y = 250
offset = 25
vehicle_count = 0
center_points = []
object_previous_y = {}  # Track previous Y-position of each object ID
counted_ids = set()     # To avoid double-counting

# Get object center
def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

# Mouse click event
def mouse_callback(event, x, y, flags, param):
    global start_processing, running
    global drawing_roi, roi_defined, roi_start, roi_end

    if event == cv2.EVENT_LBUTTONDOWN:
        if 50 <= x <= 200 and 50 <= y <= 100:
            print("[INFO] Start button clicked!")
            start_processing = True
        elif 250 <= x <= 400 and 50 <= y <= 100:
            print("[INFO] Exit button clicked!")
            running = False
        else:
            drawing_roi = True
            roi_start = (x, y)
            roi_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_roi:
            roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_roi:
            roi_end = (x, y)
            drawing_roi = False
            roi_defined = True
            print(f"[INFO] ROI defined from {roi_start} to {roi_end}")

# Set callback
cv2.namedWindow("Traffic Analysis")
cv2.setMouseCallback("Traffic Analysis", mouse_callback)
# Initialize default size thresholds
size_config = {
    "SMALL_AREA": 550,
    "MEDIUM_AREA": 4000
}

# Trackbar callback (no action needed)
def nothing(x):
    pass

# Create a control window with sliders
cv2.namedWindow("Controls")
cv2.createTrackbar("Motor Max Size", "Controls", size_config["SMALL_AREA"], 2000, nothing)
cv2.createTrackbar("Car Max Size", "Controls", size_config["MEDIUM_AREA"], 10000, nothing)

tracker = CentroidTracker()
csv_file = open('vehicle_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Vehicle ID', 'Type', 'Area', 'Center X', 'Center Y', 'Image File'])

while running:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistency
    frame = cv2.resize(frame, (640, 360))

    # Read values from sliders
    size_config["SMALL_AREA"] = cv2.getTrackbarPos("Motor Max Size", "Controls")
    size_config["MEDIUM_AREA"] = cv2.getTrackbarPos("Car Max Size", "Controls")

    # Ensure logic remains valid (Car threshold must be > Motor)
    if size_config["MEDIUM_AREA"] <= size_config["SMALL_AREA"]:
        size_config["MEDIUM_AREA"] = size_config["SMALL_AREA"] + 1
    
    # Draw detection area boundary
    #cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 255), 2)
    #cv2.putText(frame, "Detection Area", (ROI_TOP_LEFT[0], ROI_TOP_LEFT[1] - 10),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw Start/Exit buttons
    cv2.rectangle(frame, (50, 50), (200, 100), (0, 200, 0), -1)
    cv2.putText(frame, "Start", (85, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.rectangle(frame, (250, 50), (400, 100), (0, 0, 200), -1)
    cv2.putText(frame, "Exit", (290, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Draw ROI box while dragging
    if drawing_roi:
        cv2.rectangle(frame, roi_start, roi_end, (0, 255, 255), 2)

    # Final ROI box
    if roi_defined:
        top_left = (min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1]))
        bottom_right = (max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1]))

        # Draw horizontal line across the ROI
        line_y = int((top_left[1] + bottom_right[1]) / 2)  # Midpoint Y
        cv2.line(frame, (top_left[0], line_y), (bottom_right[0], line_y), (0, 0, 255), 2)

        # Optionally, label it
        cv2.putText(frame, "Counting Line", (top_left[0], line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        top_left = (min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1]))
        bottom_right = (max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)
        cv2.putText(frame, "Detection Area", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
    if start_processing:
        fg_mask = back_sub.apply(frame)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_center_points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            center = get_center(x, y, w, h)

            if roi_defined:
                roi_top_left = (min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1]))
                roi_bottom_right = (max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1]))

                if not (roi_top_left[0] <= center[0] <= roi_bottom_right[0] and
                        roi_top_left[1] <= center[1] <= roi_bottom_right[1]):
                    continue
            else:
                continue  # skip detection if ROI not defined yet            
            new_center_points.append(center)

            # Label size
            if area < size_config["SMALL_AREA"]:
            #if area < SMALL_AREA:
                label = 'Motor'
                color = (255, 0, 0)
            elif area < size_config["MEDIUM_AREA"]:
            #elif area < MEDIUM_AREA:
                label = 'Car'
                color = (0, 255, 255)
            else:
                label = 'Lorry_Van'
                color = (0, 255, 0)

            # Count when crossing line
            objects = tracker.update(new_center_points)
            for objectID, centroid in objects.items():
                current_y = centroid[1]

                # Define line if using ROI
                if roi_defined:
                    counting_line_y = int((roi_start[1] + roi_end[1]) / 2)

                previous_y = object_previous_y.get(objectID, None)

                if previous_y is not None:
                    # Detect crossing from top to bottom
                    if previous_y < counting_line_y and current_y >= counting_line_y:
                        if objectID not in counted_ids:
                            vehicle_count += 1
                            counted_ids.add(objectID)

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            image_filename = f"{output_dir}/{label}_{objectID}_{int(area)}.jpg"
                            vehicle_img = frame[y:y+h+10, x:x+w+10]
                            cv2.imwrite(image_filename, vehicle_img)

                            csv_writer.writerow([timestamp, objectID, label, int(area), centroid[0], centroid[1], image_filename])

                object_previous_y[objectID] = current_y  # Update position                           
            # Draw
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            cv2.putText(frame, f'{label} ({int(area)})', (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display count
        cv2.putText(frame, f'Vehicles Count: {vehicle_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Optionally draw line
        # cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (255, 0, 0), 2)

    # Show
    cv2.imshow("Traffic Analysis", frame)
    if start_processing:
        cv2.imshow("Foreground Mask", fg_mask)

    # Exit on ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()
