"""
Default Configuration Values

Centralized default configuration for the traffic analyzer.
"""

from typing import Dict, Tuple

DEFAULT_CONFIG = {
    # Detection
    'yolo_model': 'yolov8s.pt',
    'conf_threshold': 0.25,
    'iou_threshold': 0.45,
    'device': None,  # Auto-select
    
    # Motion detection
    'motion_history': 500,
    'motion_var_threshold': 25,
    'detect_shadows': True,
    
    # Tracking
    'tracker_max_distance': 50,
    'tracker_max_disappeared': 10,
    
    # ROI and counting
    'counting_line_y': 250,
    'counting_direction': 'down',  # 'down', 'up', 'both'
    'line_debounce_pixels': 5,
    
    # Vehicle classification
    'size_thresholds': {
        'Class 1': (60, 5),
        'Class 2': (100, 120),
        'Class 3': (120, 180),
        'Class 4': (180, 1000),
        'Class 5': (120, 300),
        'Class 6': (0, 60),
    },
    'use_six_class': True,
    
    # Perspective transform (speed estimation)
    'perspective_source': None,  # Will be set from config file
    'perspective_target_width': 25,
    'perspective_target_height': 250,
    'pixels_per_meter': 0.5,
    'time_scale_factor': 1.0,
    
    # Output
    'output_dir': 'vehicle_captures',
    'save_crops': True,
    'save_snapshots': False,
    
    # Logging
    'log_individual': True,
    'log_summary': True,
    'summary_bucket_minutes': 1,
    
    # MQTT
    'mqtt_enabled': False,
    'mqtt_broker': 'localhost',
    'mqtt_port': 1883,
    'mqtt_username': None,
    'mqtt_password': None,
    'mqtt_topic_prefix': 'traffic/data',
    
    # Video streaming
    'stream_enabled': False,
    'stream_url': None,
    'stream_use_nvenc': False,
}

