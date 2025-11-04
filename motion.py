import cv2
import time
import os
import numpy as np

# Callback function for mouse event
def mouse_callback(event, x, y, flags, param):
    global running

    # Check if left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check Start button area
        if 50 <= x <= 200 and 50 <= y <= 100:
            print("Start button clicked!")
            # Example action: draw on canvas
            cv2.putText(frame, "Started!", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Check Exit button area
        elif 250 <= x <= 400 and 50 <= y <= 100:
            print("Exit button clicked!")
            running = False
            
# Load video file
video_path = 'output.mp4'
cap = cv2.VideoCapture(video_path)

# Create background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

# Object size thresholds (in pixels)
SMALL_AREA = 1000
MEDIUM_AREA = 3000
frame_count = 0
total_count = 0
start_time = time.time()
counting_line_y = 250

offset = 25  # Allowable error in y position for counting
vehicle_count = 0
center_points = []

def get_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 360))
    fg_mask = back_sub.apply(frame)

    cv2.putText(frame, f'Vehicles Count: ', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # Draw buttons
    cv2.rectangle(frame, (50, 50), (200, 100), (0, 200, 0), -1)
    cv2.putText(frame, "Start", (85, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.rectangle(frame, (250, 50), (400, 100), (0, 0, 200), -1)
    cv2.putText(frame, "Exit", (290, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Set up window and mouse callback
    #cv2.namedWindow("Button Window")
    #cv2.setMouseCallback("Button Window", mouse_callback)

    # Morphological operations to clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (255, 0, 0), 2)
    new_center_points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue  # Ignore small noise

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        center = get_center(x, y, w, h)
        new_center_points.append(center)
        
        timestamp = time.time()
        elapsed_time = timestamp - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Draw rectangle and center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        
        # Classify based on area
        if area < SMALL_AREA:
            label = 'Small Object'
            color = (255, 0, 0)
        elif area < MEDIUM_AREA:
            label = 'Medium Object'
            color = (0, 255, 255)
        else:
            label = 'Large Object'
            color = (0, 255, 0)

        # Check if any center point crossed the line
        for center in new_center_points:
            if center not in center_points:
                if abs(center[1] - counting_line_y) <= offset:
                    vehicle_count += 1
                    center_points.append(center)

        # Display count
        cv2.putText(frame, f'              : {vehicle_count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f'{total_count} {label} ({int(area)})', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show result
    cv2.imshow('Traffic Analysis', frame)
    cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    

cap.release()
cv2.destroyAllWindows()
