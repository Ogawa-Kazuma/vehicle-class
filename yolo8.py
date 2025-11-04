from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog

# ---------------------------
# Choose video from file dialog
# ---------------------------
root = tk.Tk()
root.withdraw()  # Hide main Tk window
video_path = filedialog.askopenfilename(
    title="Select Traffic Video",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("No video selected. Exiting...")
    exit()

# ---------------------------
# Load YOLOv8 model
# ---------------------------
model = YOLO("yolov8m.pt")  # Change to yolov8n.pt, yolov8s.pt, etc.

# ---------------------------
# Custom size thresholds (width x height in pixels)
# You can tweak these based on your video resolution
# ---------------------------
SIZE_THRESHOLDS = {
    "motorcycle": (0, 60, 0, 60),     # minW, maxW, minH, maxH
    "car":        (61, 120, 61, 120),
    "van":        (121, 160, 121, 160),
    "lorry":      (161, 200, 161, 200),
    "bus":        (201, 9999, 201, 9999)
}

# ---------------------------
# Open video
# ---------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# ---------------------------
# Vehicle counts
# ---------------------------
vehicle_counts = {k: 0 for k in SIZE_THRESHOLDS.keys()}

# ---------------------------
# Process frames
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if label in ["car", "truck", "bus", "motorcycle"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1

                # Classify by custom size thresholds
                type_detected = None
                for vtype, (minW, maxW, minH, maxH) in SIZE_THRESHOLDS.items():
                    if minW <= w <= maxW and minH <= h <= maxH:
                        type_detected = vtype
                        vehicle_counts[vtype] += 1
                        break

                # Draw detection
                if type_detected:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{type_detected} ({w}x{h})",
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

    # Display vehicle counts on frame
    y_offset = 20
    for vtype, count in vehicle_counts.items():
        cv2.putText(frame, f"{vtype}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        y_offset += 25

    cv2.imshow("YOLOv8 Traffic Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
