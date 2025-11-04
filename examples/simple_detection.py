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
    model_path = 'yolov8s.pt'
    detector = YOLODetector(model_path=model_path, conf_threshold=0.25)
    
    # Check model.names to detect if it's custom or COCO
    # Custom models have names like {0: 'Class 1', 1: 'Class 2', ...}
    # COCO models have names like {0: 'person', 1: 'bicycle', 2: 'car', ...}
    model_names = detector.model.names
    is_custom_model = False
    
    # Check if model outputs classes 0-5 with "Class" names
    if len(model_names) <= 10:  # Custom models usually have 6 classes
        # Check if first few names contain "Class" or match custom pattern
        first_names = [model_names.get(i, '') for i in range(min(6, len(model_names)))]
        if any('Class' in name or 'Kelas' in name for name in first_names if name):
            is_custom_model = True
        # Also check if model names match custom pattern: Class 1, Class 2, etc.
        elif len(model_names) == 6 and all(str(i+1) in model_names.get(i, '') or 'Class' in model_names.get(i, '') for i in range(6)):
            is_custom_model = True
    
    # If using custom model, we need to map directly
    if is_custom_model:
        print(f"[INFO] Using custom model: {model_path}")
        print(f"[INFO] Model class names: {dict(model_names)}")
        print("[INFO] Model outputs classes 0-5 directly (Kelas 1-6)")
    else:
        print(f"[INFO] Using COCO model: {model_path}")
        print(f"[INFO] Model class names (sample): {dict(list(model_names.items())[:10])}")
    
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
            # For custom models, detect all classes 0-5; for COCO, use default [2,3,5,7]
            detect_classes = None if not is_custom_model else list(range(6))
            detections = detector.detect(frame, classes=detect_classes)
            
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

