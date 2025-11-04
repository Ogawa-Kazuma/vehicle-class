# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Public release preparation
- Comprehensive documentation
- Example configuration files
- MIT License
- Release checklist and guidelines

### Changed
- Organized project structure
- Archived old/duplicate script versions
- Updated requirements.txt with version constraints

### Fixed
- Code organization and cleanup

## [1.0.0] - 2024-01-01

### Added
- Interactive traffic video analyzer with YOLOv8 detection
- MOG2 motion detection integration
- Centroid-based object tracking
- Direction-aware vehicle counting
- ROI (Region of Interest) polygon selection
- Speed estimation using perspective transformation
- RTSP/RTMP streaming support via FFmpeg
- MQTT integration for real-time data publishing
- CSV logging (per-event and per-minute summaries)
- Snapshot and restore functionality
- GPU acceleration support (CUDA)
- Vehicle classification (cars, motorcycles, buses, trucks)

### Features
- Real-time video processing from files or RTSP streams
- Customizable counting lines and ROI
- Time-gated analysis windows
- Export cropped vehicle images
- Configuration persistence
