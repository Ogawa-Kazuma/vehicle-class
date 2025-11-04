# Quick Reference - Traffic Analyzer

## Installation

```bash
pip install -r requirements.txt
```

## Video File Analysis

### Quick Start (GUI File Picker)
```bash
python examples/full_analyzer.py
```
1. Select video file from dialog
2. Click "Poly" → Define ROI by clicking points
3. Click "Line" → Set counting line
4. Click "Start" → Processing begins
5. Press 'q' or click "Exit" to stop

### Simple Detection (No Tracking)
```bash
python examples/simple_detection.py
# Enter video path when prompted
```

## Live Stream Analysis

### Webcam/USB Camera
```bash
python examples/live_stream_analyzer.py
# Enter: 0 (or camera index)
```

### RTSP Stream
```bash
python examples/live_stream_analyzer.py
# Enter: rtsp://username:password@ip:port/stream
```

### Direct Python Usage
```python
from traffic_analyzer.io import VideoCapture
from traffic_analyzer.detection import YOLODetector

# Camera
with VideoCapture(0) as cap:
    detector = YOLODetector('yolov8s.pt')
    # ... process frames

# RTSP
with VideoCapture("rtsp://...") as cap:
    detector = YOLODetector('yolov8s.pt')
    # ... process frames
```

## Output Locations

- **Logs**: `vehicle_captures/{video_name}_log.csv`
- **Summary**: `vehicle_captures/{video_name}_summary_log.csv`
- **Images**: `vehicle_captures/images/`
- **Config**: `configs/{video_name}.json`

## Key Controls

| Key/Action | Function |
|------------|----------|
| **Poly** button | Define ROI polygon |
| **Line** button | Set counting line |
| **Start** button | Begin processing |
| **Exit** button | Stop and save |
| **q** key | Quit immediately |

## Common Issues

### Video won't open
- Check file path (use absolute path)
- Verify format (mp4, avi, mov, mkv)

### Stream not connecting
- Verify RTSP URL format
- Check network/credentials
- Try camera index instead

### Low FPS
- Use smaller model: `yolov8n.pt`
- Enable GPU if available
- Reduce resolution

## Module Imports

```python
# Core
from traffic_analyzer.core import AppState, VehicleClassifier

# Detection
from traffic_analyzer.detection import YOLODetector, MotionDetector

# Tracking
from traffic_analyzer.tracking import CentroidTracker

# Geometry
from traffic_analyzer.geometry import ROIManager, CountingLine

# I/O
from traffic_analyzer.io import VideoCapture, VehicleLogger, MQTTClient

# UI
from traffic_analyzer.ui import DrawingUtils, ButtonManager
```

## Configuration

```python
from traffic_analyzer.config import ConfigLoader

loader = ConfigLoader()
config = loader.load('video.mp4')  # Load saved config
loader.save('video.mp4', config)    # Save config
```

## See Also

- `USAGE_GUIDE.md` - Detailed usage instructions
- `MODULE_OVERVIEW.md` - Architecture overview
- `REFACTORING_PLAN.md` - Migration guide
- `traffic_analyzer/README.md` - Module documentation

