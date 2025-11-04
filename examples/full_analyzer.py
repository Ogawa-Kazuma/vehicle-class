#!/usr/bin/env python3
"""
Full Traffic Analyzer Example

Complete traffic analysis with detection, tracking, counting, logging, and MQTT.
Demonstrates the full capabilities of the modular traffic analyzer.
"""

import sys
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_analyzer.core import AppState, VehicleClassifier, TimeSynchronizer
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.tracking import CentroidTracker
from traffic_analyzer.geometry import ROIManager, CountingLine, CrossingDetector
from traffic_analyzer.io import VideoCapture, VehicleLogger, ImageSaver
from traffic_analyzer.ui import DrawingUtils, ButtonManager, MouseHandler
from traffic_analyzer.config import ConfigLoader


def main():
    # File selection
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    root.destroy()
    
    if not video_path:
        print("No video selected")
        return
    
    # Initialize state
    state = AppState()
    config_loader = ConfigLoader()
    config = config_loader.load(video_path)
    
    # Initialize components
    detector = YOLODetector(
        model_path=config['yolo_model'],
        conf_threshold=config['conf_threshold'],
        iou_threshold=config['iou_threshold']
    )
    tracker = CentroidTracker(
        max_distance=config['tracker_max_distance'],
        max_disappeared=config['tracker_max_disappeared']
    )
    classifier = VehicleClassifier(
        size_thresholds=config['size_thresholds'],
        use_six_class=config['use_six_class']
    )
    roi_manager = ROIManager()
    counting_line = CountingLine(initial_y=config['counting_line_y'])
    crossing_detector = CrossingDetector(debounce_pixels=config['line_debounce_pixels'])
    drawer = DrawingUtils()
    button_manager = ButtonManager()
    
    # Initialize I/O
    video_name = Path(video_path).stem
    state.output_dir = config['output_dir']
    logger = VehicleLogger(state.output_dir, video_name)
    image_saver = ImageSaver(state.output_dir, video_name)
    logger.open_logs()
    
    # Time synchronization
    time_sync = TimeSynchronizer()
    
    # Mouse handler callbacks
    def on_button_click(button_name, state):
        if button_name == "Start":
            state.start_processing = True
            roi_manager.finalize_polygon()
            if counting_line.mode == 'AUTO':
                counting_line.set_auto(roi_manager.get_polygon_mid_y())
        elif button_name == "Exit":
            state.running = False
        elif button_name == "Poly":
            state.roi.poly_editing = not state.roi.poly_editing
            if not state.roi.poly_editing:
                roi_manager.finalize_polygon()
                roi_manager.rebuild_mask((state.height, state.width, 3))
        elif button_name == "Line":
            state.counting_line.edit_mode = True
    
    def on_roi_click(point, state):
        roi_manager.rebuild_mask((state.height, state.width, 3))
    
    def on_line_click(y, state):
        state.counting_line.edit_mode = False
    
    mouse_handler = MouseHandler(
        button_manager,
        on_roi_click=on_roi_click,
        on_line_click=on_line_click,
        on_button_click=on_button_click
    )
    
    # Open video
    with VideoCapture(video_path) as cap:
        state.width = cap.width
        state.height = cap.height
        
        cv2.namedWindow("Traffic Analyzer")
        cv2.setMouseCallback(
            "Traffic Analyzer",
            mouse_handler.create_callback("Traffic Analyzer", state)
        )
        
        frame_number = 0
        
        while state.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            display = frame.copy()
            
            # Update time
            current_time = time_sync.calculate_current_time(frame_number, cap.fps)
            video_time_str = time_sync.format_video_time(frame_number, cap.fps)
            
            # Draw UI
            drawer.draw_polygon(display, roi_manager.polygon_points)
            drawer.draw_counting_line(display, counting_line.line_y)
            button_manager.draw(display)
            
            # Processing
            if state.start_processing and roi_manager.polygon_defined:
                # Detect
                detections = detector.detect(frame)
                
                # Filter by ROI
                filtered_detections = []
                centroids = []
                for det in detections:
                    if roi_manager.point_in_polygon(det['center']):
                        filtered_detections.append(det)
                        centroids.append(det['center'])
                
                # Track
                tracked_objects = tracker.update(centroids)
                
                # Process tracked objects
                for obj_id, centroid in tracked_objects.items():
                    # Find corresponding detection
                    det = None
                    for d in filtered_detections:
                        if d['center'] == centroid:
                            det = d
                            break
                    
                    if not det:
                        continue
                    
                    # Classify
                    custom_class = classifier.map_coco_to_custom(
                        det['class_id'],
                        det['width'],
                        det['height']
                    )
                    
                    if not custom_class:
                        continue
                    
                    # Update tracking state
                    if obj_id not in state.tracking.track_info:
                        state.tracking.track_info[obj_id] = {
                            'last_centroid': None,
                            'counted': False,
                            'last_y_bottom': None
                        }
                    
                    track_info = state.tracking.track_info[obj_id]
                    prev_y = track_info['last_y_bottom']
                    curr_y = det['bbox'][3]  # Bottom Y coordinate
                    
                    # Check crossing
                    if not track_info['counted']:
                        if crossing_detector.crossed_line(
                            prev_y, curr_y, counting_line.line_y,
                            state.count_direction
                        ):
                            track_info['counted'] = True
                            state.vehicle_counts.counted_ids.add(obj_id)
                            state.vehicle_counts.by_class[custom_class] += 1
                            
                            # Log
                            image_path = ""
                            if config['save_crops']:
                                image_path = image_saver.save_crop(
                                    frame, det['bbox'], obj_id, custom_class, current_time
                                )
                            
                            logger.log_vehicle(
                                current_time, obj_id, custom_class,
                                det['confidence'], video_time_str, image_path
                            )
                    
                    track_info['last_centroid'] = centroid
                    track_info['last_y_bottom'] = curr_y
                    
                    # Draw
                    color = classifier.get_class_color(custom_class)
                    drawer.draw_bbox(
                        display, det['bbox'], color,
                        f"ID:{obj_id} {custom_class}"
                    )
                    
                    # Draw trail
                    if obj_id in state.tracking.tracking_trails:
                        state.tracking.tracking_trails[obj_id].append(centroid)
                        if len(state.tracking.tracking_trails[obj_id]) > 30:
                            state.tracking.tracking_trails[obj_id].pop(0)
                    else:
                        state.tracking.tracking_trails[obj_id] = [centroid]
                    
                    drawer.draw_trail(display, state.tracking.tracking_trails[obj_id], color)
                
                # Update summary log
                if time_sync.should_update_summary(current_time):
                    logger.log_summary(current_time, state.vehicle_counts.by_class)
            
            # Draw counts
            drawer.draw_counts(display, state.vehicle_counts.by_class)
            drawer.draw_info(display, {
                'Frame': str(frame_number),
                'Time': video_time_str,
                'Status': 'PROCESSING' if state.start_processing else 'STANDBY'
            })
            
            # Display
            cv2.imshow("Traffic Analyzer", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Save config
        config_loader.save_roi(video_path, roi_manager.polygon_points)
        config_loader.save_counting_line(video_path, counting_line.line_y, counting_line.mode)
        
        # Close logs
        logger.close()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

