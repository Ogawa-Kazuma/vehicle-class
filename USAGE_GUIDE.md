# Traffic Analyzer - Usage Guide

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually:
pip install ultralytics opencv-python numpy paho-mqtt supervision
```

### Basic Usage with Video File

#### Method 1: Using the Full Analyzer Example

```bash
# Run the full analyzer with GUI file picker
python examples/full_analyzer.py
```

**Steps:**
1. A file dialog will open - select your video file
2. The video window will open with controls:
   - **Poly** button: Click to start defining ROI polygon (click points on frame)
   - **Line** button: Click to set counting line position (click on frame)
   - **Start** button: Begin processing
   - **Exit** button: Stop and save
   - **Press 'q'**: Quit anytime

3. **Workflow:**
   ```
   a) Click "Poly" → Click points on frame to define ROI → Click "Start" to finalize
   b) Click "Line" → Click on frame to set counting line → Automatically set
   c) Click "Start" → Processing begins
   d) Watch detections and counts in real-time
   e) Click "Exit" or press 'q' to stop
   ```

4. **Output:**
   - Logs saved to: `vehicle_captures/{video_name}_log.csv`
   - Summary saved to: `vehicle_captures/{video_name}_summary_log.csv`
   - Images saved to: `vehicle_captures/images/`
   - Config saved to: `configs/{video_name}.json`

#### Method 2: Using Simple Detection (No Tracking/Counting)

```bash
python examples/simple_detection.py
```

**Steps:**
1. Enter video path when prompted
2. View detections in real-time
3. Press 'q' to quit

#### Method 3: Programmatic Usage

```python
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.io.video import VideoCapture
from traffic_analyzer.ui.drawing import DrawingUtils

# Initialize
detector = YOLODetector(model_path='yolov8s.pt', conf_threshold=0.25)
drawer = DrawingUtils()

# Process video
with VideoCapture('path/to/video.mp4') as cap:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        detections = detector.detect(frame)
        
        # Draw results
        for det in detections:
            drawer.draw_bbox(
                frame,
                det['bbox'],
                label=f"{det['class_name']} {det['confidence']:.2f}"
            )
        
        # Display
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

### Live Stream Usage

#### Method 1: Camera Input (Webcam/USB Camera)

```python
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.io.video import VideoCapture
from traffic_analyzer.ui.drawing import DrawingUtils

# Initialize
detector = YOLODetector(model_path='yolov8s.pt')
drawer = DrawingUtils()

# Open camera (0 = default camera, 1 = second camera, etc.)
with VideoCapture(0) as cap:
    print(f"Camera: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        detections = detector.detect(frame)
        
        # Draw and display
        for det in detections:
            drawer.draw_bbox(frame, det['bbox'], 
                           label=f"{det['class_name']} {det['confidence']:.2f}")
        
        cv2.imshow("Live Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
```

#### Method 2: RTSP Stream

```python
from traffic_analyzer.detection import YOLODetector
from traffic_analyzer.io.video import VideoCapture
from traffic_analyzer.ui.drawing import DrawingUtils

# RTSP URL format: rtsp://username:password@ip:port/stream
rtsp_url = "rtsp://admin:password123@192.168.1.100:554/stream1"

detector = YOLODetector(model_path='yolov8s.pt')
drawer = DrawingUtils()

try:
    with VideoCapture(rtsp_url) as cap:
        print(f"Stream: {cap.width}x{cap.height} @ {cap.fps:.2f} fps")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, retrying...")
                continue
            
            # Detect vehicles
            detections = detector.detect(frame)
            
            # Draw and display
            for det in detections:
                drawer.draw_bbox(frame, det['bbox'],
                               label=f"{det['class_name']} {det['confidence']:.2f}")
            
            cv2.imshow("RTSP Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except Exception as e:
    print(f"Error: {e}")
finally:
    cv2.destroyAllWindows()
```

#### Method 3: Full Analyzer with Live Stream

Create `examples/live_stream_analyzer.py`:

```python
#!/usr/bin/env python3
"""
Live Stream Traffic Analyzer

Analyzes traffic from live camera or RTSP stream.
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
    source = input("Enter camera index (0,1,2...) or RTSP URL: ").strip()
    
    # Try to parse as integer (camera index)
    try:
        source = int(source)
    except ValueError:
        pass  # Keep as string (RTSP URL)
    
    # Initialize
    state = AppState()
    detector = YOLODetector(model_path='yolov8s.pt', conf_threshold=0.25)
    tracker = CentroidTracker()
    classifier = VehicleClassifier()
    roi_manager = ROIManager()
    counting_line = CountingLine()
    crossing_detector = CrossingDetector()
    drawer = DrawingUtils()
    button_manager = ButtonManager()
    
    # Setup logging (optional for live stream)
    video_name = f"live_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = VehicleLogger("live_logs", video_name)
    logger.open_logs()
    image_saver = ImageSaver("live_logs", video_name)
    
    # Mouse handler
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
        elif button_name == "Line":
            state.counting_line.edit_mode = True
    
    mouse_handler = MouseHandler(button_manager, on_button_click=on_button_click)
    
    # Open stream
    try:
        with VideoCapture(source) as cap:
            state.width = cap.width
            state.height = cap.height
            
            cv2.namedWindow("Live Stream Analyzer")
            cv2.setMouseCallback(
                "Live Stream Analyzer",
                mouse_handler.create_callback("Live Stream Analyzer", state)
            )
            
            frame_count = 0
            
            while state.running:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
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
                    filtered = [d for d in detections 
                               if roi_manager.point_in_polygon(d['center'])]
                    
                    # Track
                    centroids = [d['center'] for d in filtered]
                    tracked = tracker.update(centroids)
                    
                    # Process and draw
                    for obj_id, centroid in tracked.items():
                        det = next((d for d in filtered if d['center'] == centroid), None)
                        if not det:
                            continue
                        
                        custom_class = classifier.map_coco_to_custom(
                            det['class_id'], det['width'], det['height']
                        )
                        
                        if custom_class:
                            color = classifier.get_class_color(custom_class)
                            drawer.draw_bbox(display, det['bbox'], color,
                                           f"ID:{obj_id} {custom_class}")
                            
                            # Check crossing and count
                            if obj_id not in state.tracking.track_info:
                                state.tracking.track_info[obj_id] = {
                                    'last_y_bottom': None,
                                    'counted': False
                                }
                            
                            info = state.tracking.track_info[obj_id]
                            prev_y = info['last_y_bottom']
                            curr_y = det['bbox'][3]
                            
                            if not info['counted']:
                                if crossing_detector.crossed_line(
                                    prev_y, curr_y, counting_line.line_y
                                ):
                                    info['counted'] = True
                                    state.vehicle_counts.by_class[custom_class] += 1
                            
                            info['last_y_bottom'] = curr_y
                
                # Draw counts
                drawer.draw_counts(display, state.vehicle_counts.by_class)
                drawer.draw_info(display, {
                    'Frame': str(frame_count),
                    'Status': 'PROCESSING' if state.start_processing else 'STANDBY'
                })
                
                # Display
                cv2.imshow("Live Stream Analyzer", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        logger.close()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

Run it:
```bash
python examples/live_stream_analyzer.py
# Enter: 0 (for webcam) or rtsp://...
```

## Configuration

### Customizing Detection Settings

```python
from traffic_analyzer.detection import YOLODetector

# Custom model and thresholds
detector = YOLODetector(
    model_path='yolov8m.pt',  # Larger model for better accuracy
    conf_threshold=0.3,        # Lower = more detections
    iou_threshold=0.5         # Higher = fewer overlapping boxes
)
```

### Customizing Tracking

```python
from traffic_analyzer.tracking import CentroidTracker

tracker = CentroidTracker(
    max_distance=100,      # Max pixels for matching
    max_disappeared=15    # Frames before removing object
)
```

### Customizing Vehicle Classification

```python
from traffic_analyzer.core import VehicleClassifier

# Custom size thresholds
custom_thresholds = {
    'Class 1': (50, 10),
    'Class 2': (90, 110),
    # ... etc
}

classifier = VehicleClassifier(
    size_thresholds=custom_thresholds,
    use_six_class=True
)
```

## Output Files

### CSV Logs
- **Individual log**: `{video_name}_log.csv`
  - Columns: Timestamp, ID, Type, Confidence, VideoTime, Image
- **Summary log**: `{video_name}_summary_log.csv`
  - Columns: Time, Class 1-6, Total

### Images
- **Cropped vehicles**: `images/{video_name}_{timestamp}_{id}_{class}_{counter}.jpg`
- **Snapshots**: `images/{video_name}_snapshot_{timestamp}.jpg`

### Configuration
- **Saved config**: `configs/{video_name}.json`
  - ROI polygon points
  - Counting line position
  - Perspective transform points
  - All other settings

## Troubleshooting

### Video File Issues
- **Error: "Failed to open video"**
  - Check file path is correct
  - Verify video format is supported (mp4, avi, mov, mkv)
  - Try with absolute path

### Live Stream Issues
- **RTSP stream not connecting**
  - Verify URL format: `rtsp://user:pass@ip:port/stream`
  - Check network connectivity
  - Verify credentials
  - Try increasing timeout in VideoCapture

- **Camera not detected**
  - Try different camera index (0, 1, 2...)
  - Check camera permissions
  - Verify camera is not in use by another application

### Performance Issues
- **Low FPS**
  - Use smaller YOLO model (yolov8n.pt instead of yolov8s.pt)
  - Reduce input resolution
  - Use GPU if available
  - Lower confidence threshold to reduce processing

- **GPU not detected**
  - Check CUDA installation: `nvidia-smi`
  - Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
  - Install CUDA-enabled PyTorch if needed

## Advanced Usage

### MQTT Integration

```python
from traffic_analyzer.io import MQTTClient
from datetime import datetime

mqtt = MQTTClient(
    broker="broker.example.com",
    port=8883,
    username="user",
    password="pass"
)
mqtt.connect()

# Publish 15-minute report
mqtt.publish_15min_report(
    datetime.now(),
    class_counts={'Class 1': 10, 'Class 2': 5},
    avg_speed=50.5,
    avg_headway=2.3
)
```

### Video Streaming Output

```python
from traffic_analyzer.io import VideoStreamer
import numpy as np

streamer = VideoStreamer(
    output_url="rtsp://localhost:8554/live",
    width=1920,
    height=1080,
    fps=30.0,
    use_nvenc=True  # Hardware acceleration
)

streamer.start()

# In your processing loop:
streamer.write(annotated_frame)

# When done:
streamer.close()
```

### Custom ROI and Counting Line

```python
from traffic_analyzer.geometry import ROIManager, CountingLine

roi = ROIManager()
roi.add_point((100, 100))
roi.add_point((500, 100))
roi.add_point((500, 400))
roi.add_point((100, 400))
roi.finalize_polygon()
roi.rebuild_mask((height, width, 3))

counting_line = CountingLine()
counting_line.set_manual(250)  # Y coordinate
```

## Command Line Examples

```bash
# Simple detection
python examples/simple_detection.py

# Full analyzer (with GUI file picker)
python examples/full_analyzer.py

# Live stream analyzer
python examples/live_stream_analyzer.py

# With custom model
python -c "
from traffic_analyzer.detection import YOLODetector
detector = YOLODetector('yolov8m.pt')
print('Model loaded')
"
```

## Next Steps

- See `examples/` directory for more examples
- Read `MODULE_OVERVIEW.md` for architecture details
- Check `REFACTORING_PLAN.md` for migration guide
- Review `traffic_analyzer/README.md` for module documentation

