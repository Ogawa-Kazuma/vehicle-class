#!/usr/bin/env python3
"""
Simple Detection Example

Basic YOLO vehicle detection without tracking or counting.
Demonstrates minimal usage of the modular traffic analyzer.
"""

import sys
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.ui.drawing import DrawingUtils
from traffic_analyzer.io.video import VideoCapture


def main():
    # Initialize components
    detector = YOLODetector(model_path='yolov8s.pt', conf_threshold=0.25)
    drawer = DrawingUtils()
    
    # Open video
    video_path = input("Enter video path: ").strip()
    if not video_path:
        print("No video path provided")
        return
    
    with VideoCapture(video_path) as cap:
        print(f"Video: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles
            detections = detector.detect(frame)
            
            # Draw detections
            for det in detections:
                drawer.draw_bbox(
                    frame,
                    det['bbox'],
                    color=(0, 255, 0),
                    label=f"{det['class_name']} {det['confidence']:.2f}"
                )
            
            # Show counts
            counts = {}
            for det in detections:
                cls = det['class_name']
                counts[cls] = counts.get(cls, 0) + 1
            
            drawer.draw_counts(frame, counts)
            
            # Display
            cv2.imshow("Simple Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

