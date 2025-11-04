# Installation Guide

This guide provides detailed installation instructions for the Traffic Video Analyzer.

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space

### Recommended for Best Performance
- **CPU**: Multi-core processor (4+ cores)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **RAM**: 16GB or more
- **Storage**: SSD with 10GB+ free space

## Step-by-Step Installation

### 1. Prerequisites

#### Install Python

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv
```

**macOS:**
```bash
# Using Homebrew
brew install python3
```

**Windows:**
Download and install from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python3 --version  # Should show 3.8 or higher
```

#### Install FFmpeg (Optional, for streaming)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract and add to PATH

Verify installation:
```bash
ffmpeg -version
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/traffic-video-analyzer.git
cd traffic-video-analyzer
```

### 3. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. GPU Support (Optional)

For NVIDIA GPU acceleration:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (default)
# Already installed via requirements.txt
```

Verify GPU availability:
```python
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 6. Verify Installation

Run a test to verify everything works:

```bash
python3 -c "import cv2, torch, ultralytics; print('All imports successful!')"
```

## Quick Setup Scripts

### Automated Setup

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

These scripts will:
- Check Python version
- Create virtual environment
- Install all dependencies
- Verify FFmpeg installation
- Check CUDA availability

## Installation Issues

### Problem: pip install fails

**Solution:**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Problem: OpenCV installation fails

**Solution (Linux):**
```bash
sudo apt-get install python3-opencv
pip install opencv-python-headless
```

### Problem: PyTorch CUDA not working

**Solution:**
1. Verify NVIDIA driver: `nvidia-smi`
2. Install matching CUDA toolkit
3. Reinstall PyTorch with correct CUDA version

### Problem: tkinter not found (Linux)

**Solution:**
```bash
sudo apt-get install python3-tk
```

## Post-Installation

### Download YOLOv8 Models

Models download automatically on first use, or manually:

```python
from ultralytics import YOLO

# Download different model sizes
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']
for model_name in models:
    YOLO(model_name)
```

### Test Installation

Run a quick test:

```bash
python vehicle_detection.py
```

If you see the file dialog, installation is successful!

## Next Steps

- Read [Usage Guide](USAGE.md)
- Review [Configuration Guide](CONFIGURATION.md)
- Check [FAQ](FAQ.md)
