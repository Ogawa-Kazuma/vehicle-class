#!/usr/bin/env python3
"""
Headless Traffic Analyzer

Processes videos or streams in headless mode using saved configurations.
No GUI, no user interaction - perfect for server deployments.

Usage:
    python headless_analyzer.py <video_path_or_stream_url>
    
Example:
    python headless_analyzer.py /path/to/video.mp4
    python headless_analyzer.py rtsp://example.com/stream
    python headless_analyzer.py camera_0
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_analyzer.core import AppState, VehicleClassifier, TimeSynchronizer
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.tracking import CentroidTracker
from traffic_analyzer.geometry import ROIManager, CountingLine, CrossingDetector
from traffic_analyzer.io import VideoCapture, VehicleLogger, ImageSaver
from traffic_analyzer.config import ConfigLoader


def create_config_key(source):
    """
    Create a safe config key from stream source.
    
    Args:
        source: Video path or stream URL
        
    Returns:
        Config key string
    """
    if isinstance(source, str):
        # Check if it's a video file path
        if Path(source).exists() or source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return Path(source).stem
        # Check if it's a camera index reference
        elif source.startswith('camera_'):
            return source
        # Otherwise treat as URL
        else:
            # Create safe filename from URL
            safe_key = re.sub(r'^https?://', '', source)
            safe_key = re.sub(r'[:/?#\[\]@!$&\'()*+,;=]', '_', safe_key)
            safe_key = safe_key.replace('.', '_')
            if len(safe_key) > 50:
                safe_key = safe_key[:50]
            return safe_key
    else:
        return "unknown_source"


def validate_config(config, config_key, source):
    """
    Validate that required config exists.
    
    Args:
        config: Configuration dictionary
        config_key: Config key name
        source: Source path/URL
        
    Returns:
        bool: True if config is valid
    """
    roi_polygon = config.get('roi_polygon', [])
    counting_line_y = config.get('counting_line_y')
    
    if not roi_polygon or len(roi_polygon) < 3:
        print(f"[ERROR] No valid ROI polygon found in config for '{config_key}'")
        print(f"[ERROR] Please configure ROI using full_analyzer.py or live_stream_analyzer.py first")
        return False
    
    if counting_line_y is None:
        print(f"[ERROR] No counting line configured for '{config_key}'")
        print(f"[ERROR] Please configure counting line using full_analyzer.py or live_stream_analyzer.py first")
        return False
    
    print(f"[INFO] Config validated: ROI with {len(roi_polygon)} points, counting line at y={counting_line_y}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Headless Traffic Analyzer - Process videos/streams without GUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/video.mp4
  %(prog)s rtsp://example.com/stream
  %(prog)s camera_0
  
The script requires a saved configuration file. Configure ROI and counting line
first using full_analyzer.py or live_stream_analyzer.py.
        """
    )
    parser.add_argument(
        'source',
        type=str,
        help='Video file path, RTSP stream URL, or camera config key (e.g., camera_0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='headless_output',
        help='Output directory for logs and images (default: headless_output)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8s.pt',
        help='YOLO model path (default: yolov8s.pt)'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Detection confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Do not save vehicle images'
    )
    
    args = parser.parse_args()
    source = args.source
    
    print("=" * 60)
    print("Headless Traffic Analyzer")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Create config key
    config_key = create_config_key(source)
    print(f"[INFO] Using config key: {config_key}")
    
    # Initialize config loader
    config_loader = ConfigLoader()
    config = config_loader.load(config_key)
    
    # Validate config
    if not validate_config(config, config_key, source):
        print("\n[ERROR] Invalid configuration. Exiting.")
        sys.exit(1)
    
    # Initialize state
    state = AppState()
    
    # Initialize components
    model_path = args.model or config.get('yolo_model', 'yolov8s.pt')
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=args.conf_threshold or config.get('conf_threshold', 0.25),
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
    
    # Setup ROI from config
    roi_manager = ROIManager()
    saved_roi = config.get('roi_polygon', [])
    roi_manager.polygon_points = [tuple(p) if isinstance(p, (list, tuple)) else p for p in saved_roi]
    roi_manager.polygon_defined = True
    
    # Setup counting line from config
    counting_line = CountingLine(initial_y=config.get('counting_line_y', 250))
    counting_line.mode = config.get('counting_line_mode', 'MANUAL')
    if counting_line.mode == 'AUTO' and roi_manager.polygon_defined:
        counting_line.set_auto(roi_manager.get_polygon_mid_y())
    
    crossing_detector = CrossingDetector(debounce_pixels=config.get('line_debounce_pixels', 5))
    
    # Determine actual video source
    if source.startswith('camera_'):
        # Extract camera index
        try:
            camera_index = int(source.split('_')[1])
            video_source = camera_index
        except (ValueError, IndexError):
            print(f"[ERROR] Invalid camera format: {source}")
            sys.exit(1)
    else:
        video_source = source
    
    # Setup logging
    video_name = config_key
    state.output_dir = args.output_dir
    logger = VehicleLogger(state.output_dir, video_name)
    image_saver = ImageSaver(state.output_dir, video_name) if not args.no_images else None
    logger.open_logs()
    
    # Time synchronization
    time_sync = TimeSynchronizer()
    if config.get('start_date') and config.get('start_time'):
        time_sync.set_start_time(config['start_date'], config['start_time'])
    
    state.count_direction = config.get('counting_direction', 'down')
    
    # Process video/stream
    print(f"\n[INFO] Starting processing...")
    print(f"[INFO] ROI polygon: {len(roi_manager.polygon_points)} points")
    print(f"[INFO] Counting line: y={counting_line.line_y}, mode={counting_line.mode}")
    print(f"[INFO] Counting direction: {state.count_direction}")
    print()
    
    try:
        with VideoCapture(video_source) as cap:
            state.width = cap.width
            state.height = cap.height
            
            # Rebuild ROI mask
            if roi_manager.polygon_defined and len(roi_manager.polygon_points) >= 3:
                roi_manager.rebuild_mask((state.height, state.width, 3))
            
            print(f"[INFO] Stream: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")
            print(f"[INFO] Processing frames... (Press Ctrl+C to stop)\n")
            
            frame_number = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of stream reached")
                    break
                
                frame_number += 1
                
                # Update time
                current_time = time_sync.calculate_current_time(frame_number, cap.fps)
                video_time_str = time_sync.format_video_time(frame_number, cap.fps)
                
                # Detect
                # For custom models, detect all classes 0-5; for COCO, use default [2,3,5,7]
                detect_classes = None if not is_custom_model else list(range(6))
                detections = detector.detect(frame, classes=detect_classes)
                
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
                        if best_i is not None and best_d < 2500:
                            track_to_det[obj_id] = best_i
                            unused.discard(best_i)
                
                for obj_id, centroid in tracked_objects.items():
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
                    curr_y = det['bbox'][3]
                    
                    # Check crossing
                    if not track_info['counted']:
                        if crossing_detector.crossed_line(
                            prev_y, curr_y, counting_line.line_y,
                            state.count_direction
                        ):
                            track_info['counted'] = True
                            state.vehicle_counts.counted_ids.add(obj_id)
                            state.vehicle_counts.by_class[custom_class] += 1
                            processed_count += 1
                            
                            # Log
                            image_path = ""
                            if image_saver:
                                image_path = image_saver.save_crop(
                                    frame, det['bbox'], obj_id, custom_class, current_time
                                )
                            
                            logger.log_vehicle(
                                current_time, obj_id, custom_class,
                                det['confidence'], video_time_str, image_path
                            )
                            
                            # Progress output
                            if processed_count % 10 == 0 or processed_count == 1:
                                print(f"[{video_time_str}] Frame {frame_number}: "
                                      f"Vehicle #{processed_count} - {custom_class} "
                                      f"(ID: {obj_id}, Conf: {det['confidence']:.2f})")
                    
                    track_info['last_centroid'] = centroid
                    track_info['last_y_bottom'] = curr_y
                
                # Update summary log periodically
                if time_sync.should_update_summary(current_time, bucket_minutes=config.get('summary_bucket_minutes', 1)):
                    logger.log_summary(current_time, state.vehicle_counts.by_class)
                    state.vehicle_counts.by_class.clear()
                
                # Progress indicator every 100 frames
                if frame_number % 100 == 0:
                    print(f"[INFO] Processed {frame_number} frames, "
                          f"{processed_count} vehicles counted so far...")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Processing error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Final summary log
        logger.log_summary(current_time, state.vehicle_counts.by_class)
        logger.close()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Processing Summary")
        print("=" * 60)
        print(f"Total frames processed: {frame_number}")
        print(f"Total vehicles counted: {processed_count}")
        print(f"\nVehicle counts by class:")
        for vehicle_type, count in sorted(state.vehicle_counts.by_class.items()):
            print(f"  {vehicle_type}: {count}")
        print(f"\nLogs saved to: {args.output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    main()
