#!/usr/bin/env python3
"""
Live Stream Traffic Analyzer

Analyzes traffic from live camera or RTSP stream with full tracking and counting.
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_analyzer.core import AppState, VehicleClassifier
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.tracking import CentroidTracker
from traffic_analyzer.geometry import ROIManager, CountingLine, CrossingDetector
from traffic_analyzer.io import VideoCapture, VehicleLogger, ImageSaver
from traffic_analyzer.ui import DrawingUtils, ButtonManager, MouseHandler
from traffic_analyzer.config import ConfigLoader
from datetime import datetime
import re


def create_config_key(source):
    """
    Create a safe config key from stream source.
    
    Args:
        source: Camera index (int) or RTSP URL (str)
        
    Returns:
        Config key string
    """
    if isinstance(source, int):
        return f"camera_{source}"
    elif isinstance(source, str):
        # Create safe filename from URL
        # Remove protocol and special characters
        safe_key = re.sub(r'^https?://', '', source)
        safe_key = re.sub(r'[:/?#\[\]@!$&\'()*+,;=]', '_', safe_key)
        safe_key = safe_key.replace('.', '_')
        # Limit length
        if len(safe_key) > 50:
            safe_key = safe_key[:50]
        return safe_key
    else:
        return "unknown_source"


def main():
    # Get stream source
    print("Live Stream Traffic Analyzer")
    print("=" * 40)
    source_input = input("Enter camera index (0,1,2...) or RTSP URL (or press Enter for camera 0): ").strip()
    
    if not source_input:
        source = 0
    else:
        # Try to parse as integer (camera index)
        try:
            source = int(source_input)
        except ValueError:
            source = source_input  # Keep as string (RTSP URL)
    
    # Create config key for this source
    config_key = create_config_key(source)
    print(f"[INFO] Config key: {config_key}")
    
    # Initialize config loader
    config_loader = ConfigLoader()
    
    # Load saved config (using config_key as virtual video path)
    config = config_loader.load(config_key)
    
    # Initialize
    state = AppState()
    model_path = config.get('yolo_model', 'yolov8s.pt')
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=config.get('conf_threshold', 0.25),
        iou_threshold=config.get('iou_threshold', 0.45)
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
        max_distance=config.get('tracker_max_distance', 50),
        max_disappeared=config.get('tracker_max_disappeared', 10)
    )
    classifier = VehicleClassifier(
        size_thresholds=config.get('size_thresholds'),
        use_six_class=config.get('use_six_class', True)
    )
    roi_manager = ROIManager()
    counting_line = CountingLine()
    crossing_detector = CrossingDetector(debounce_pixels=config.get('line_debounce_pixels', 5))
    drawer = DrawingUtils()
    button_manager = ButtonManager()
    
    # Restore saved ROI polygon if available
    saved_roi = config.get('roi_polygon', [])
    if saved_roi and len(saved_roi) >= 3:
        # Convert to tuples if needed
        roi_manager.polygon_points = [tuple(p) if isinstance(p, (list, tuple)) else p for p in saved_roi]
        roi_manager.polygon_defined = True
        print(f"[INFO] Loaded saved ROI polygon with {len(roi_manager.polygon_points)} points")
    else:
        print("[INFO] No saved ROI polygon found, starting fresh")
    
    # Restore counting line if available
    saved_line_y = config.get('counting_line_y')
    if saved_line_y is not None:
        counting_line.line_y = saved_line_y
    saved_line_mode = config.get('counting_line_mode', 'AUTO')
    counting_line.mode = saved_line_mode
    if saved_line_mode == 'AUTO' and roi_manager.polygon_defined:
        counting_line.set_auto(roi_manager.get_polygon_mid_y())
        print(f"[INFO] Restored counting line mode: AUTO (y={counting_line.line_y})")
    else:
        print(f"[INFO] Restored counting line: y={counting_line.line_y}, mode={saved_line_mode}")
    
    # Setup logging
    video_name = f"live_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    state.output_dir = "live_logs"
    logger = VehicleLogger(state.output_dir, video_name)
    logger.open_logs()
    image_saver = ImageSaver(state.output_dir, video_name)
    
    # Mouse handler callbacks
    def on_button_click(button_name, state):
        if button_name == "Start":
            state.start_processing = True
            roi_manager.finalize_polygon()
            if counting_line.mode == 'AUTO' and roi_manager.polygon_defined:
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
                if state.width and state.height:
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
        if len(roi_manager.polygon_points) >= 3 and state.width and state.height:
            roi_manager.rebuild_mask((state.height, state.width, 3))
        print(f"[INFO] Added point {point}, total points: {len(roi_manager.polygon_points)}")
    
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
    
    # Open stream
    try:
        with VideoCapture(source) as cap:
            state.width = cap.width
            state.height = cap.height
            
            # Sync restored ROI with state and rebuild mask if polygon is defined
            if roi_manager.polygon_defined and len(roi_manager.polygon_points) >= 3:
                state.roi.polygon_points = roi_manager.polygon_points.copy()
                state.roi.polygon_defined = True
                roi_manager.rebuild_mask((state.height, state.width, 3))
                print(f"[INFO] Rebuilt ROI mask for stream dimensions: {state.width}x{state.height}")
            
            # Sync counting line with state
            state.counting_line.line_y = counting_line.line_y
            state.count_direction = config.get('counting_direction', 'down')
            
            print(f"Stream opened: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")
            print("Controls:")
            print("  - Click 'Poly' button, then click points on frame to define ROI")
            print("  - Click 'Line' button, then click on frame to set counting line")
            print("  - Click 'Start' to begin processing")
            print("  - Press 'q' to quit")
            
            cv2.namedWindow("Live Stream Analyzer", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(
                "Live Stream Analyzer",
                mouse_handler.create_callback("Live Stream Analyzer", state)
            )
            
            frame_count = 0
            current_time = datetime.now()
            
            while state.running:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, retrying...")
                    continue
                
                frame_count += 1
                display = frame.copy()
                
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
                
                # Processing
                if state.start_processing and roi_manager.polygon_defined:
                    # For custom models, detect all classes 0-5; for COCO, use default [2,3,5,7]
                    detect_classes = None if not is_custom_model else list(range(6))
                    detections = detector.detect(frame, classes=detect_classes)
                    
                    # Filter by ROI
                    filtered = []
                    centroids = []
                    for det in detections:
                        if roi_manager.point_in_polygon(det['center']):
                            filtered.append(det)
                            centroids.append(det['center'])
                    
                    # Track
                    tracked = tracker.update(centroids)
                    
                    # Process tracked objects
                    for obj_id, centroid in tracked.items():
                        # Find corresponding detection
                        det = None
                        for d in filtered:
                            if d['center'] == centroid:
                                det = d
                                break
                        
                        if not det:
                            continue
                        
                        # Classify - handle both COCO and custom models
                        if is_custom_model:
                            # Custom model outputs 0-5 directly, map to Class names
                            class_map_custom = {0: 'Class 1', 1: 'Class 2', 2: 'Class 3', 
                                               3: 'Class 4', 4: 'Class 5', 5: 'Class 6'}
                            custom_class = class_map_custom.get(det['class_id'])
                        else:
                            # COCO model: map COCO classes to custom classes via size
                            custom_class = classifier.map_coco_to_custom(
                                det['class_id'], det['width'], det['height']
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
                                try:
                                    image_path = image_saver.save_crop(
                                        frame, det['bbox'], obj_id, custom_class, current_time
                                    )
                                    logger.log_vehicle(
                                        current_time, obj_id, custom_class,
                                        det['confidence'], f"Frame:{frame_count}", image_path
                                    )
                                except Exception as e:
                                    print(f"Error saving: {e}")
                        
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
                
                # Draw counts
                drawer.draw_counts(display, state.vehicle_counts.by_class)
                drawer.draw_info(display, {
                    'Frame': str(frame_count),
                    'Status': 'PROCESSING' if state.start_processing else 'STANDBY',
                    'Time': current_time.strftime("%H:%M:%S")
                })
                
                # Display
                cv2.imshow("Live Stream Analyzer", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Update time every second
                if frame_count % int(cap.fps) == 0:
                    current_time = datetime.now()
        
        # Save final summary
        logger.log_summary(current_time, state.vehicle_counts.by_class)
        logger.close()
        
        # Save config (only if polygon is defined)
        if roi_manager.polygon_defined and len(roi_manager.polygon_points) >= 3:
            config_loader.save_roi(config_key, roi_manager.polygon_points)
            print(f"[INFO] Saved ROI polygon configuration")
        config_loader.save_counting_line(config_key, counting_line.line_y, counting_line.mode)
        print(f"[INFO] Saved counting line configuration")
        
        print(f"\nProcessing complete!")
        print(f"Total vehicles counted: {sum(state.vehicle_counts.by_class.values())}")
        print(f"Logs saved to: {state.output_dir}/")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

