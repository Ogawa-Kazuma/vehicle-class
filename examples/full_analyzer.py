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
    # Check if model is custom (outputs 0-5 directly) or COCO (outputs 2,3,5,7)
    model_path = config['yolo_model']
    
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=config['conf_threshold'],
        iou_threshold=config['iou_threshold']
    )
    
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
        print("[INFO] Will map COCO classes (2,3,5,7) to custom classes")
    tracker = CentroidTracker(
        max_distance=config['tracker_max_distance'],
        max_disappeared=config['tracker_max_disappeared']
    )
    classifier = VehicleClassifier(
        size_thresholds=config['size_thresholds'],
        use_six_class=config['use_six_class']
    )
    roi_manager = ROIManager()
    
    # Restore saved ROI polygon if available
    saved_roi = config.get('roi_polygon', [])
    if saved_roi and len(saved_roi) >= 3:
        # Convert to tuples if needed
        roi_manager.polygon_points = [tuple(p) if isinstance(p, (list, tuple)) else p for p in saved_roi]
        roi_manager.polygon_defined = True
        print(f"[INFO] Loaded saved ROI polygon with {len(roi_manager.polygon_points)} points")
    else:
        print("[INFO] No saved ROI polygon found, starting fresh")
    
    counting_line = CountingLine(initial_y=config['counting_line_y'])
    # Restore counting line mode if available
    saved_line_mode = config.get('counting_line_mode', 'MANUAL')
    counting_line.mode = saved_line_mode
    if saved_line_mode == 'AUTO' and roi_manager.polygon_defined:
        counting_line.set_auto(roi_manager.get_polygon_mid_y())
        print(f"[INFO] Restored counting line mode: AUTO (y={counting_line.line_y})")
    else:
        print(f"[INFO] Restored counting line: y={counting_line.line_y}, mode={saved_line_mode}")
    
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
    if config.get('start_date') and config.get('start_time'):
        time_sync.set_start_time(config['start_date'], config['start_time'])
    
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
            # Toggle poly editing mode
            new_value = not roi_manager.poly_editing
            roi_manager.poly_editing = new_value
            state.roi.poly_editing = new_value  # Sync state
            # Also sync with state.roi_manager if it exists
            if hasattr(state, 'roi_manager'):
                state.roi_manager.poly_editing = new_value
            print(f"[DEBUG] Poly button clicked: roi_manager.poly_editing = {roi_manager.poly_editing}")
            print(f"[DEBUG] After sync: state.roi_manager.poly_editing = {getattr(state.roi_manager, 'poly_editing', 'N/A') if hasattr(state, 'roi_manager') else 'roi_manager not in state'}")
            if roi_manager.poly_editing:
                # Starting polygon editing - clear previous points
                roi_manager.reset()
                # Don't clear poly_editing in reset, we just set it!
                roi_manager.poly_editing = True  # Re-set after reset
                if hasattr(state, 'roi_manager'):
                    state.roi_manager.poly_editing = True
                state.roi.poly_editing = True
                state.roi.polygon_points = []
                print("[INFO] Polygon edit mode: left-click to add points, right-click or 'p' to close.")
            else:
                # Ending polygon editing
                roi_manager.finalize_polygon()
                state.roi.polygon_defined = roi_manager.polygon_defined
                state.roi.polygon_points = roi_manager.polygon_points.copy()
                if state.height and state.width:
                    roi_manager.rebuild_mask((state.height, state.width, 3))
                print(f"[INFO] Polygon defined with {len(roi_manager.polygon_points)} points")
        elif button_name == "Line":
            state.counting_line.edit_mode = True
            counting_line.edit_mode = True
            print("[INFO] Line edit mode: click anywhere to set counting line (press 'A' for AUTO).")
    
    def on_roi_click(point, state):
        print(f"[DEBUG] on_roi_click called with point {point}")
        # Add point to both roi_manager and state
        roi_manager.add_point(point)
        state.roi.polygon_points.append(point)
        # Only rebuild mask if we have enough points, otherwise just draw points
        if len(roi_manager.polygon_points) >= 3:
            roi_manager.rebuild_mask((state.height, state.width, 3))
        print(f"[INFO] Added point {point}, total points: {len(roi_manager.polygon_points)}")
        print(f"[DEBUG] roi_manager.polygon_points: {roi_manager.polygon_points}")
        print(f"[DEBUG] state.roi_manager.polygon_points: {getattr(state.roi_manager, 'polygon_points', 'N/A')}")
    
    def on_line_click(y, state):
        state.counting_line.edit_mode = False
        counting_line.edit_mode = False
        print(f"[INFO] Counting line set to y={counting_line.line_y}")
    
    # Attach managers to state for mouse handler access (before creating callback)
    state.roi_manager = roi_manager
    state.counting_line_obj = counting_line
    
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
        
        # Sync restored ROI with state and rebuild mask if polygon is defined
        if roi_manager.polygon_defined and len(roi_manager.polygon_points) >= 3:
            state.roi.polygon_points = roi_manager.polygon_points.copy()
            state.roi.polygon_defined = True
            roi_manager.rebuild_mask((state.height, state.width, 3))
            print(f"[INFO] Rebuilt ROI mask for video dimensions: {state.width}x{state.height}")
        
        # Sync counting line with state
        state.counting_line.line_y = counting_line.line_y
        state.count_direction = config.get('counting_direction', 'down')
        
        print(f"[DEBUG] Attached roi_manager to state, poly_editing = {roi_manager.poly_editing}")
        print(f"[DEBUG] state.roi_manager exists: {hasattr(state, 'roi_manager')}")
        if hasattr(state, 'roi_manager'):
            print(f"[DEBUG] state.roi_manager.poly_editing = {state.roi_manager.poly_editing}")
        
        cv2.namedWindow("Traffic Analyzer", cv2.WINDOW_NORMAL)
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
            # Draw polygon with filled overlay if defined, otherwise just outline
            if roi_manager.polygon_defined and len(roi_manager.polygon_points) >= 3:
                # Draw filled overlay
                overlay = display.copy()
                drawer.draw_polygon(overlay, roi_manager.polygon_points, filled=True)
                display = cv2.addWeighted(overlay, 0.15, display, 0.85, 0)
            
            # Always draw polygon outline and points (even during editing)
            if len(roi_manager.polygon_points) > 0:
                drawer.draw_polygon(display, roi_manager.polygon_points, filled=False)
            
            drawer.draw_counting_line(display, counting_line.line_y)
            button_manager.draw(display)
            
            # Debug info
            if roi_manager.poly_editing:
                cv2.putText(display, f"Polygon Editing: {len(roi_manager.polygon_points)} points", 
                           (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 255), 2)
            
            # Always detect (even in standby) to show what's being detected
            # For custom models, detect all classes 0-5; for COCO, use default [2,3,5,7]
            detect_classes = None if not is_custom_model else list(range(6))
            detections = detector.detect(frame, classes=detect_classes)
            
            # Debug: Print detection count on first frame and periodically
            if frame_number == 1:
                print(f"[INFO] Detection initialized. Frame 1: {len(detections)} detections found")
                # Debug: Show what classes are detected
                for det in detections[:3]:  # Show first 3
                    print(f"  - {det['class_name']} (ID:{det['class_id']}) size: {det['width']}x{det['height']} conf:{det['confidence']:.2f}")
            elif frame_number % 100 == 0 and len(detections) > 0:
                print(f"[INFO] Frame {frame_number}: {len(detections)} detections found")
                # Debug: Show classifications
                # Use the detected model type from initialization
                is_custom = is_custom_model
                
                for det in detections[:2]:
                    if is_custom_model:
                        class_map_custom = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3', 
                                           3: 'Class 4', 4: 'Class 5', 5: 'Class 6'}
                        test_class = class_map_custom.get(det['class_id'])
                    else:
                        test_class = classifier.map_coco_to_custom(det['class_id'], det['width'], det['height'])
                    print(f"  - {det['class_name']} (ID:{det['class_id']}) {det['width']}x{det['height']} -> {test_class} (custom_model={is_custom_model})")
            
            # Track which detections are tracked (to avoid drawing duplicates)
            # Use bbox as key since it's unique per detection
            tracked_detections = set()
            
            # Processing
            if state.start_processing and roi_manager.polygon_defined:
                # Filter by ROI
                filtered_detections = []
                centroids = []
                for det in detections:
                    if roi_manager.point_in_polygon(det['center']):
                        filtered_detections.append(det)
                        centroids.append(det['center'])
                
                # Debug output every 30 frames
                if frame_number % 30 == 0:
                    print(f"[DEBUG] Frame {frame_number}: Total detections: {len(detections)}, "
                          f"Filtered by ROI: {len(filtered_detections)}")
                
                # Track
                tracked_objects = tracker.update(centroids)
                
                # Process tracked objects
                # Build mapping from centroids to detections
                track_to_det = {}
                if filtered_detections:
                    det_centers = [d['center'] for d in filtered_detections]
                    unused = set(range(len(filtered_detections)))
                    for obj_id, trk_centroid in tracked_objects.items():
                        best_i, best_d = None, float("inf")
                        for i in list(unused):
                            dcx, dcy = det_centers[i]
                            d = (trk_centroid[0] - dcx)**2 + (trk_centroid[1] - dcy)**2
                            if d < best_d:
                                best_d, best_i = d, i
                        if best_i is not None and best_d < 2500:  # Max distance threshold (50 pixels squared)
                            track_to_det[obj_id] = best_i
                            unused.discard(best_i)
                
                for obj_id, centroid in tracked_objects.items():
                    # Find corresponding detection
                    det_idx = track_to_det.get(obj_id, None)
                    if det_idx is None:
                        continue
                    
                    det = filtered_detections[det_idx]
                    
                    # Classify - handle both COCO and custom models
                    if is_custom_model:
                        # Custom model outputs 0-5 directly, map to Class names
                        class_map_custom = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3', 
                                           3: 'Class 4', 4: 'Class 5', 5: 'Class 6'}
                        custom_class = class_map_custom.get(det['class_id'])
                    else:
                        # COCO model: map COCO classes to custom classes via size
                        custom_class = classifier.map_coco_to_custom(
                            det['class_id'],
                            det['width'],
                            det['height']
                        )
                    
                    # Debug: Log classification failures
                    if not custom_class and frame_number % 30 == 0:
                        print(f"[DEBUG] Classification failed: class {det['class_id']} ({det['class_name']}) "
                              f"size {det['width']}x{det['height']}, is_custom={is_custom_model}")
                    
                    # Mark this detection as tracked (only if we're going to process it)
                    # Even if classification fails, we still mark it so it's not drawn as untracked
                    tracked_detections.add(det['bbox'])
                    
                    # Draw tracked detection even if classification fails (with default label)
                    if not custom_class:
                        # For COCO cars that are too small, default to Class 1
                        if not is_custom_model and det['class_id'] == 2:  # COCO car
                            custom_class = 'Class 1'
                            print(f"[INFO] Small car detected ({det['width']}px), assigning to Class 1")
                        else:
                            # Draw with default label if classification fails for other types
                            color = (128, 128, 128)  # Gray for unclassified
                            drawer.draw_bbox(
                                display, det['bbox'], color,
                                f"ID:{obj_id} {det['class_name']} ({det['width']}x{det['height']})"
                            )
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
                    
                    # Draw tracked detection with ID and trail
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
                    
                    # Debug: Confirm we drew this tracked detection
                    if frame_number % 30 == 0:
                        print(f"[DEBUG] Drew tracked detection ID:{obj_id} - {custom_class} at {det['bbox']}")
                
                # Update summary log
                if time_sync.should_update_summary(current_time, bucket_minutes=config.get('summary_bucket_minutes', 1)):
                    logger.log_summary(current_time, state.vehicle_counts.by_class)
                    state.vehicle_counts.by_class.clear()  # Reset counts after logging summary
            
            # Draw untracked detections (for visibility)
            # In standby: show all detections (even outside ROI for setup)
            # In processing: show only untracked ones within ROI
            untracked_count = 0
            car_detections = 0
            class1_detections = 0
            if len(detections) > 0:
                for det in detections:
                    # Track car detections for debugging
                    if det['class_id'] == 2:  # COCO car
                        car_detections += 1
                    
                    # Skip if already drawn as tracked
                    if det['bbox'] in tracked_detections:
                        continue
                    
                    # In processing mode: filter by ROI if polygon is defined
                    # In standby mode: show all detections (to help with ROI setup)
                    if state.start_processing and roi_manager.polygon_defined:
                        if not roi_manager.point_in_polygon(det['center']):
                            continue
                    
                    # Classify for label - handle both COCO and custom models
                    if is_custom_model:
                        # Custom model outputs 0-5 directly
                        class_map_custom = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3', 
                                           3: 'Class 4', 4: 'Class 5', 5: 'Class 6'}
                        label_class = class_map_custom.get(det['class_id'])
                    else:
                        # COCO model: map via classifier
                        label_class = classifier.map_coco_to_custom(
                            det['class_id'],
                            det['width'],
                            det['height']
                        )
                        # Default small cars to Class 1
                        if not label_class and det['class_id'] == 2:
                            label_class = 'Class 1'
                            if frame_number % 30 == 0:
                                print(f"[DEBUG] Untracked car ({det['width']}x{det['height']}) defaulted to Class 1")
                    
                    # Track Class 1 detections
                    if label_class == 'Class 1':
                        class1_detections += 1
                        if frame_number % 30 == 0:
                            print(f"[DEBUG] Class 1 vehicle detected: {det['class_name']} {det['width']}x{det['height']} at {det['center']}")
                    
                    # Draw with appropriate color
                    if state.start_processing:
                        # In processing mode, untracked detections in gray
                        color = (128, 128, 128)
                        if label_class:
                            label = f"{label_class} {det['confidence']:.2f} (untracked)"
                        else:
                            label = f"{det['class_name']} {det['confidence']:.2f} (untracked)"
                    else:
                        # In standby mode, show all detections
                        # Use different color if outside ROI (for setup guidance)
                        if roi_manager.polygon_defined and not roi_manager.point_in_polygon(det['center']):
                            color = (100, 100, 100)  # Darker gray for outside ROI
                        else:
                            if label_class == 'Class 1':
                                color = (255, 255, 0)  # Cyan for Class 1 (more visible than white)
                            elif label_class == 'Class 6':
                                color = (0, 255, 255)  # Yellow for Class 6
                            else:
                                color = (0, 255, 100)  # Light green for other classes
                        
                        if label_class:
                            label = f"{label_class} {det['confidence']:.2f}"
                        else:
                            label = f"{det['class_name']} {det['width']}x{det['height']} {det['confidence']:.2f}"
                    
                    drawer.draw_bbox(display, det['bbox'], color, label)
                    untracked_count += 1
            
            # Debug output for drawing
            if frame_number % 30 == 0:
                print(f"[DEBUG] Drawing: {len(tracked_detections)} tracked, {untracked_count} untracked drawn, "
                      f"processing={state.start_processing}, roi_defined={roi_manager.polygon_defined}")
                if car_detections > 0:
                    print(f"[DEBUG] Car detections: {car_detections} total, {class1_detections} classified as Class 1")
            
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
        
        # Save config (only if polygon is defined)
        if roi_manager.polygon_defined and len(roi_manager.polygon_points) >= 3:
            config_loader.save_roi(video_path, roi_manager.polygon_points)
        config_loader.save_counting_line(video_path, counting_line.line_y, counting_line.mode)
        
        # Close logs
        logger.close()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

