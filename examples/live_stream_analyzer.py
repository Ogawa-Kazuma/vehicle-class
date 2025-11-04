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
from datetime import datetime


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
    
    # Initialize
    state = AppState()
    detector = YOLODetector(model_path='yolov8s.pt', conf_threshold=0.25)
    tracker = CentroidTracker(max_distance=50, max_disappeared=10)
    classifier = VehicleClassifier()
    roi_manager = ROIManager()
    counting_line = CountingLine()
    crossing_detector = CrossingDetector(debounce_pixels=5)
    drawer = DrawingUtils()
    button_manager = ButtonManager()
    
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
            state.roi.poly_editing = not state.roi.poly_editing
            if not state.roi.poly_editing:
                roi_manager.finalize_polygon()
                if state.width and state.height:
                    roi_manager.rebuild_mask((state.height, state.width, 3))
        elif button_name == "Line":
            state.counting_line.edit_mode = True
    
    def on_roi_click(point, state):
        if state.width and state.height:
            roi_manager.rebuild_mask((state.height, state.width, 3))
    
    def on_line_click(y, state):
        state.counting_line.edit_mode = False
    
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
            
            print(f"Stream opened: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")
            print("Controls:")
            print("  - Click 'Poly' button, then click points on frame to define ROI")
            print("  - Click 'Line' button, then click on frame to set counting line")
            print("  - Click 'Start' to begin processing")
            print("  - Press 'q' to quit")
            
            cv2.namedWindow("Live Stream Analyzer")
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
                drawer.draw_polygon(display, roi_manager.polygon_points)
                drawer.draw_counting_line(display, counting_line.line_y)
                button_manager.draw(display)
                
                # Processing
                if state.start_processing and roi_manager.polygon_defined:
                    detections = detector.detect(frame)
                    
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
                        
                        # Classify
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

