# Traffic Analyzer - Modular Architecture

## Overview

This is the refactored, modular version of the traffic analyzer codebase. It consolidates 37+ Python scripts into a clean, maintainable structure with clear separation of concerns.

## Structure

```
traffic_analyzer/
├── core/              # Core functionality
│   ├── state.py       # Application state management
│   ├── vehicle_classifier.py  # Vehicle classification
│   └── time_sync.py   # Time synchronization
├── tracking/          # Object tracking
│   ├── centroid_tracker.py  # Centroid-based tracking
│   └── bytetrack_wrapper.py  # ByteTrack integration
├── detection/         # Vehicle detection
│   ├── yolo_detector.py     # YOLO detection
│   ├── motion_detector.py   # Motion-based detection
│   └── hybrid_detector.py   # Combined approach
├── geometry/          # Geometry and ROI
│   ├── roi.py        # ROI management
│   ├── counting_line.py  # Counting line
│   ├── crossing.py   # Line crossing detection
│   └── perspective.py  # Perspective transform
├── io/               # Input/Output
│   ├── video.py      # Video capture/streaming
│   ├── logger.py     # CSV logging
│   ├── mqtt_client.py  # MQTT publishing
│   └── image_saver.py  # Image saving
├── ui/               # User interface
│   ├── drawing.py    # Drawing utilities
│   ├── buttons.py    # Button management
│   └── mouse_handler.py  # Mouse event handling
├── config/           # Configuration
│   ├── defaults.py   # Default values
│   └── loader.py     # Config file I/O
└── utils/            # Utilities
    ├── device.py     # Device selection
    ├── file_utils.py # File utilities
    └── validation.py # Input validation
```

## Usage Examples

### Simple Detection

```python
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.io.video import VideoCapture

detector = YOLODetector(model_path='yolov8s.pt')
with VideoCapture('video.mp4') as cap:
    ret, frame = cap.read()
    detections = detector.detect(frame)
```

### Full Analysis

See `examples/full_analyzer.py` for a complete implementation.

## Key Benefits

1. **Modularity**: Each component is self-contained (100-300 lines)
2. **Reusability**: Components can be used independently
3. **Testability**: Each module can be unit tested
4. **Maintainability**: Clear module boundaries
5. **Documentation**: Well-documented interfaces

## Migration from Old Scripts

The old scripts in the root directory can be gradually migrated to use the new modular structure. Each old script maps to a combination of modules:

- `motion_v1.py`, `motion_v2.py` → `detection.motion_detector` + `tracking.centroid_tracker`
- `detection_yolo8_*.py` → `detection.yolo_detector` + `tracking.centroid_tracker`
- `mix_mog2nyolo8_*.py` → `detection.hybrid_detector` + `tracking.centroid_tracker`
- `kkr_vcd.py`, `vehicle_detection.py` → Full stack (all modules)

## Next Steps

1. Create example scripts for each use case
2. Write unit tests for each module
3. Migrate existing scripts to use new modules
4. Update documentation

