# Traffic Video Analyzer

An interactive AI-powered traffic video analyzer that detects, tracks, and counts vehicles using YOLOv8 object detection and motion analysis. Perfect for traffic monitoring, vehicle counting, and speed estimation from video feeds or IP camera streams.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)

## ✨ Features

- 🚗 **Vehicle Detection**: Detects cars, motorcycles, buses, and trucks using YOLOv8
- 📊 **Direction-Aware Counting**: Counts vehicles crossing a configurable counting line (up, down, or both directions)
- 🎯 **Interactive ROI Selection**: Draw polygon regions of interest for focused analysis
- 🔄 **Object Tracking**: Centroid-based tracking for consistent vehicle identification
- 📹 **Multiple Input Sources**: Supports local video files and RTSP/IP camera streams
- ⚡ **Motion Detection**: MOG2 background subtraction to filter static objects
- 📈 **Speed Estimation**: Calculate vehicle speeds using perspective transformation
- 📝 **CSV Logging**: Per-event and per-minute summary logs
- 🖼️ **Image Export**: Saves cropped images of detected vehicles
- 📡 **RTSP/RTMP Streaming**: Stream annotated frames via FFmpeg (optional GPU acceleration)
- 🔌 **MQTT Integration**: Real-time data publishing for IoT systems
- 💾 **Snapshot & Restore**: Save/load configuration for resumable analysis
- ⏰ **Time Gating**: Analyze specific time windows within videos
- 🎮 **GPU Support**: CUDA acceleration when available

## 📋 Requirements

- Python 3.8 or higher
- OpenCV 4.9+
- PyTorch 2.0+
- Ultralytics YOLOv8
- FFmpeg (optional, for streaming)
- CUDA-compatible GPU (optional, for acceleration)

## 🚀 Quick Start

### Installation

#### Option 1: Automated Setup (Recommended)

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

#### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/traffic-video-analyzer.git
cd traffic-video-analyzer
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. (Optional) Install FFmpeg for streaming:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### Download Models

The application requires YOLOv8 models. Models will be automatically downloaded on first run, or you can download manually:

```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')  # Downloads automatically
```

Available models: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`

## 💻 Usage

### Basic Usage

Run the main application:
```bash
python vehicle_detection.py
```

The application will:
1. Prompt you to select a video file or enter an RTSP URL
2. Display the video with interactive controls
3. Allow you to draw ROI and counting lines
4. Process and count vehicles automatically

### MQTT Integration

For MQTT-enabled version:
```bash
python mqtt.py
```

Configure MQTT settings in the script or via environment variables.

### Training Custom Models

To train on your own dataset:
```bash
python training.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Click` | Draw ROI polygon points |
| `P` or `Right-click` | Close ROI polygon |
| `L` | Set counting line manually |
| `A` | Auto-set counting line (mid-ROI) |
| `U` | Set counting direction UP |
| `D` | Set counting direction DOWN |
| `B` | Set counting direction BOTH |
| `R` | Reset polygon |
| `C` | Capture snapshot |
| `T` | Toggle streaming (if configured) |
| `ESC` | Exit application |

## ⚙️ Configuration

### Configuration File

Create a configuration file (JSON format) for speed estimation calibration:

```json
{
    "perspective_points": [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
    ],
    "run_date": "2024-01-01",
    "run_start_time": "08:00:00"
}
```

Example configuration available in `examples/configs/config_example.json`.

### Calibrating Speed Estimation

1. Identify four points in your video frame that form a rectangle in the real world
2. Update `perspective_points` in your config file
3. Adjust `PIXELS_PER_METER_SCALE` in the script for accuracy
4. Set `TIME_SCALE_FACTOR` if video playback speed differs from real-time

## 📁 Project Structure

```
traffic-video-analyzer/
├── vehicle_detection.py    # Main application
├── mqtt.py                 # MQTT-enabled version
├── training.py             # Model training script
├── requirements.txt        # Python dependencies
├── setup.sh / setup.bat    # Setup scripts
├── LICENSE                 # MIT License
├── README.md               # This file
├── CHANGELOG.md            # Version history
├── examples/               # Example files
│   ├── configs/           # Configuration templates
│   ├── videos/            # Sample videos (small)
│   └── models/            # Model download links
├── docs/                   # Additional documentation
└── archive/                # Old/archived scripts
```

## 📊 Output

The application generates:

- **CSV Logs**: Per-event detection logs (`*_log.csv`)
- **Summary Logs**: Per-minute aggregated counts (`*_summary_log.csv`)
- **Vehicle Images**: Cropped images saved to `vehicle_captures/<video_name>/`
- **Snapshots**: Configuration snapshots with overlay images

## 🔧 Advanced Features

### GPU Acceleration

For CUDA support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Streaming Output

Configure streaming in the script:
- RTSP: `rtsp://your-server:554/stream`
- RTMP: `rtmp://your-server:1935/live/stream`

Press `T` to toggle streaming during runtime.

### Custom Vehicle Classes

Modify the detection classes in the script:
```python
classes=[2,3,5,7]  # COCO IDs: car=2, motorcycle=3, bus=5, truck=7
```

## 🐛 Troubleshooting

### Common Issues

**Problem**: "No module named 'cv2'"
- **Solution**: Install OpenCV: `pip install opencv-python`

**Problem**: "CUDA out of memory"
- **Solution**: Use a smaller YOLO model (yolov8n.pt) or reduce input resolution

**Problem**: FFmpeg streaming not working
- **Solution**: Verify FFmpeg installation and RTSP/RTMP server accessibility

**Problem**: Slow performance
- **Solution**: Enable GPU acceleration or reduce video resolution

### Getting Help

- Check existing issues on GitHub
- Review the documentation in `docs/`
- Open a new issue with details about your problem

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision
- [Supervision](https://github.com/roboflow/supervision) for tracking utilities
- All contributors and users of this project

## 📚 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- [MQTT Protocol](https://mqtt.org/)

## 🗺️ Roadmap

- [ ] Web interface for remote access
- [ ] Real-time dashboard
- [ ] Additional vehicle classification
- [ ] License plate recognition integration
- [ ] Database integration for long-term storage
- [ ] Docker containerization

## ⚠️ Known Limitations

- Speed estimation requires manual calibration
- Object tracking may have ID switches in crowded scenes
- Processing speed depends on hardware capabilities
- Large videos may require significant memory

## 📧 Contact

For questions, suggestions, or support, please open an issue on GitHub.

---

**Made with ❤️ for traffic analysis**
