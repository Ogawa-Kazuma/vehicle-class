# Scripts Documentation

This document describes the main scripts in the Traffic Video Analyzer project.

## Main Application Scripts

### `vehicle_detection.py`
**Main entry point for the traffic video analyzer application.**

Features:
- Interactive GUI for video analysis
- ROI polygon selection
- Counting line setup
- Real-time vehicle detection and tracking
- CSV logging
- Snapshot and restore functionality
- RTSP/RTMP streaming support (via FFmpeg)

Usage:
```bash
python vehicle_detection.py
```

### `mqtt.py`
**MQTT-enabled version with speed estimation and real-time data publishing.**

Features:
- All features from `vehicle_detection.py`
- MQTT integration for IoT systems
- Speed estimation using perspective transformation
- Real-time data publishing

Usage:
```bash
python mqtt.py
```

Configuration:
- Set MQTT broker details in script or via environment variables
- Configure perspective points for speed estimation

## Training Scripts

### `training.py`
**Train custom YOLOv8 models on your dataset.**

Usage:
```bash
python training.py
```

Requirements:
- Organized dataset in YOLO format
- Configuration file (`data.yaml`)

## Utility Scripts

### `detection_yolo8.py`
**Basic YOLOv8 detection script (simplified version).**

Use this for quick detection tests without full GUI.

### `annotate.py`
**Annotation tool for creating training datasets.**

Helps label vehicles in videos for custom model training.

### `roboflow.py`
**Integration with Roboflow for dataset management.**

Work with Roboflow datasets and upload/download annotations.

### `kkr_vcd.py`
**Vehicle classification detection script.**

Additional classification and detection utilities.

### `image_sampling.py`
**Extract sample frames from videos.**

Useful for creating training datasets or analysis samples.

### `generate_veh_cls.py` / `make_veh_cls_from_video.py`
**Generate vehicle classification datasets from videos.**

Extract and organize vehicle images for classification training.

### `mix_mog2nyolo8.py`
**Combines MOG2 motion detection with YOLOv8.**

Experimental integration of motion detection with object detection.

### `motion.py`
**Motion detection using MOG2.**

Standalone motion detection for background subtraction.

### `yolo8.py`
**Simple YOLOv8 detection demo.**

Minimal example for YOLOv8 usage.

## Archived Scripts

The following scripts have been moved to `archive/` directory:
- `detection_yolo8_alpha.py`
- `detection_yolo8_beta.py`
- `detection_yolo8_v1.py`
- `mix_mog2nyolo8_v1.py`
- `mix_mog2nyolo8_v2.py`
- `motion_v1.py`
- `motion_v2.py`
- `roboflow (Copy).py`
- `testing_v12.py`

These are older versions kept for reference but not recommended for use.

## Script Selection Guide

**For end users:**
- Use `vehicle_detection.py` for standard traffic analysis
- Use `mqtt.py` if you need IoT integration or speed estimation

**For developers:**
- Use utility scripts for specific tasks
- Check `training.py` for custom model development

**For dataset creation:**
- Use `annotate.py` for labeling
- Use `image_sampling.py` for frame extraction
- Use `generate_veh_cls.py` for classification datasets

## Adding New Scripts

When adding new scripts:

1. Add clear docstring at the top
2. Include usage instructions in docstring
3. Follow project coding standards
4. Update this documentation
5. Add to appropriate category

Example script header:
```python
"""
Script Name: Short description

Purpose:
Detailed description of what the script does.

Usage:
    python script_name.py [options]

Requirements:
    - List any special requirements

Author: Your Name
Date: YYYY-MM-DD
"""
```
