# Traffic Analyzer - Module Overview

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (examples/full_analyzer.py, examples/simple_detection.py)  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   AppState   │  │ Classifier  │  │  TimeSync    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Detection    │  │   Tracking    │  │   Geometry    │
│  ┌─────────┐  │  │  ┌─────────┐  │  │  ┌─────────┐  │
│  │ YOLO    │  │  │  │Centroid │  │  │  │   ROI   │  │
│  │ Motion  │  │  │  │ByteTrack│  │  │  │  Line   │  │
│  │ Hybrid  │  │  │  └─────────┘  │  │  │Crossing │  │
│  └─────────┘  │  └───────────────┘  │  │Perspect │  │
└───────────────┘                      └───────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    I/O Layer                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Video   │  │  Logger  │  │   MQTT   │  │  Image   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    UI Layer                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │ Drawing  │  │ Buttons │  │  Mouse   │                    │
│  └──────────┘  └──────────┘  └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Support Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │ Config   │  │  Device │  │  Utils   │                    │
│  └──────────┘  └──────────┘  └──────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
Core (State, Classifier, TimeSync)
  ├── Used by: All modules
  └── Dependencies: Standard library only

Detection (YOLO, Motion, Hybrid)
  ├── Used by: Application layer
  └── Dependencies: ultralytics, opencv-python, torch

Tracking (CentroidTracker, ByteTrack)
  ├── Used by: Application layer
  ├── Dependencies: numpy, supervision (optional)
  └── Input: Detection results

Geometry (ROI, Line, Crossing, Perspective)
  ├── Used by: Detection, Application layer
  └── Dependencies: opencv-python, numpy

I/O (Video, Logger, MQTT, Image)
  ├── Used by: Application layer
  └── Dependencies: opencv-python, paho-mqtt (optional)

UI (Drawing, Buttons, Mouse)
  ├── Used by: Application layer
  └── Dependencies: opencv-python

Config & Utils
  ├── Used by: Application layer
  └── Dependencies: Standard library
```

## File Size Distribution

| Module Category | Files | Avg Lines | Total Lines |
|----------------|-------|-----------|-------------|
| Core           | 3     | ~150      | ~450        |
| Tracking       | 2     | ~100      | ~200        |
| Detection      | 3     | ~200      | ~600        |
| Geometry       | 4     | ~150      | ~600        |
| I/O            | 4     | ~200      | ~800        |
| UI             | 3     | ~150      | ~450        |
| Config         | 2     | ~100      | ~200        |
| Utils          | 3     | ~50       | ~150        |
| **Total**      | **24**| **~140**  | **~3,450**  |

## Comparison: Old vs New

### Old Structure
```
Root directory:
  ├── motion_v1.py (300 lines)
  ├── motion_v2.py (350 lines)
  ├── detection_yolo8_v1.py (400 lines)
  ├── detection_yolo8_alpha.py (1200 lines)
  ├── mix_mog2nyolo8_v1.py (800 lines)
  ├── mix_mog2nyolo8_v2.py (850 lines)
  ├── mix_mog2nyolo8.py (1000 lines)
  ├── kkr_vcd.py (1500 lines)
  ├── vehicle_detection.py (1400 lines)
  └── ... (28 more files)
  
Issues:
  - Code duplication across files
  - Hard to maintain
  - Difficult to test
  - Inconsistent patterns
```

### New Structure
```
traffic_analyzer/
  ├── core/ (3 focused modules)
  ├── tracking/ (2 focused modules)
  ├── detection/ (3 focused modules)
  ├── geometry/ (4 focused modules)
  ├── io/ (4 focused modules)
  ├── ui/ (3 focused modules)
  ├── config/ (2 focused modules)
  └── utils/ (3 focused modules)

Benefits:
  - Single source of truth for each component
  - Easy to maintain and test
  - Consistent interfaces
  - Reusable across projects
```

## Usage Patterns

### Pattern 1: Simple Detection Only
```python
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.io.video import VideoCapture

detector = YOLODetector('yolov8s.pt')
# Use detector.detect(frame)
```

### Pattern 2: Detection + Tracking
```python
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.tracking import CentroidTracker

detector = YOLODetector('yolov8s.pt')
tracker = CentroidTracker()
# Use detector + tracker together
```

### Pattern 3: Full Analysis
```python
from traffic_analyzer.core import AppState
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.tracking import CentroidTracker
from traffic_analyzer.geometry import ROIManager, CountingLine
from traffic_analyzer.io import VideoCapture, VehicleLogger
# Use all components together
```

## Migration Checklist

- [x] Create folder structure
- [x] Implement core modules
- [x] Implement tracking modules
- [x] Implement detection modules
- [x] Implement geometry modules
- [x] Implement I/O modules
- [x] Implement UI modules
- [x] Implement config modules
- [x] Implement utils modules
- [x] Create example scripts
- [x] Write documentation
- [ ] Write unit tests
- [ ] Migrate existing scripts
- [ ] Performance testing
- [ ] Update main README

## Quick Reference

### Import All Modules
```python
from traffic_analyzer.core import AppState, VehicleClassifier, TimeSynchronizer
from traffic_analyzer.detection import YOLODetector, MotionDetector, HybridDetector
from traffic_analyzer.tracking import CentroidTracker, ByteTrackWrapper
from traffic_analyzer.geometry import ROIManager, CountingLine, CrossingDetector, PerspectiveTransformer
from traffic_analyzer.io import VideoCapture, VideoStreamer, VehicleLogger, MQTTClient, ImageSaver
from traffic_analyzer.ui import DrawingUtils, ButtonManager, MouseHandler
from traffic_analyzer.config import ConfigLoader, DEFAULT_CONFIG
from traffic_analyzer.utils import select_device, make_safe_filename, validate_video_path
```

### Common Workflow
1. Initialize state: `state = AppState()`
2. Load config: `config = ConfigLoader().load(video_path)`
3. Initialize detectors/trackers
4. Open video: `with VideoCapture(video_path) as cap:`
5. Process frames in loop
6. Save config: `ConfigLoader().save(video_path, config)`

