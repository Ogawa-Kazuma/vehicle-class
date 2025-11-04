# Refactoring Plan: Traffic Analyzer Codebase Consolidation

## Overview
This document outlines the plan to consolidate 37+ Python scripts into a modular, maintainable codebase structure.

## Current Issues
1. **Code Duplication**: CentroidTracker, polygon handling, ROI management repeated across 10+ files
2. **Inconsistent Patterns**: Different approaches to same functionality (motion_v1, motion_v2, detection_yolo8 variants)
3. **Hardcoded Values**: Configuration scattered throughout scripts
4. **Large Monolithic Files**: Some scripts exceed 1000 lines
5. **Poor Separation of Concerns**: UI, logic, and I/O mixed together

## New Structure

```
traffic_analyzer/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── state.py              # Global state management
│   ├── vehicle_classifier.py # Class mapping (6-class system)
│   └── time_sync.py          # Time synchronization utilities
├── tracking/
│   ├── __init__.py
│   ├── centroid_tracker.py   # CentroidTracker implementation
│   └── bytetrack_wrapper.py # ByteTrack integration
├── detection/
│   ├── __init__.py
│   ├── yolo_detector.py      # YOLO model wrapper
│   ├── motion_detector.py    # Background subtraction
│   └── hybrid_detector.py    # Combined motion + YOLO
├── geometry/
│   ├── __init__.py
│   ├── roi.py                # ROI polygon management
│   ├── counting_line.py      # Counting line logic
│   ├── perspective.py      # Perspective transform
│   └── crossing.py           # Line crossing detection
├── io/
│   ├── __init__.py
│   ├── video.py              # Video capture/streaming
│   ├── logger.py             # CSV logging (individual + summary)
│   ├── mqtt_client.py        # MQTT publishing
│   └── image_saver.py        # Crop saving functionality
├── ui/
│   ├── __init__.py
│   ├── drawing.py            # Drawing utilities (bboxes, trails, etc.)
│   ├── buttons.py             # UI button management
│   └── mouse_handler.py      # Mouse event callbacks
├── config/
│   ├── __init__.py
│   ├── loader.py             # Config file loading/saving
│   └── defaults.py           # Default configuration values
└── utils/
    ├── __init__.py
    ├── device.py             # GPU/CPU selection
    ├── file_utils.py         # File path utilities
    └── validation.py         # Input validation

examples/
├── simple_detection.py        # Basic YOLO detection
├── motion_based.py            # Motion detection only
├── hybrid_analysis.py        # Motion + YOLO hybrid
├── full_analyzer.py          # Complete traffic analyzer
└── dataset_generator.py      # Video to dataset converter

tests/
├── test_tracking.py
├── test_detection.py
├── test_geometry.py
└── test_io.py
```

## Migration Strategy

### Phase 1: Core Components (Week 1)
1. Extract `CentroidTracker` → `tracking/centroid_tracker.py`
2. Extract YOLO model loading → `detection/yolo_detector.py`
3. Extract ROI/polygon logic → `geometry/roi.py`
4. Extract configuration → `config/loader.py`

### Phase 2: I/O Components (Week 2)
1. Extract CSV logging → `io/logger.py`
2. Extract MQTT client → `io/mqtt_client.py`
3. Extract video handling → `io/video.py`
4. Extract image saving → `io/image_saver.py`

### Phase 3: UI & Utilities (Week 3)
1. Extract drawing functions → `ui/drawing.py`
2. Extract mouse handlers → `ui/mouse_handler.py`
3. Extract utility functions → `utils/`

### Phase 4: Integration (Week 4)
1. Create main application entry point
2. Create example scripts using new modules
3. Update documentation
4. Test with existing video files

## Key Benefits

1. **Modularity**: Each component in separate file (100-300 lines)
2. **Reusability**: Components can be imported and used independently
3. **Testability**: Each module can be unit tested
4. **Maintainability**: Changes isolated to specific modules
5. **Documentation**: Clear module boundaries and responsibilities

## Backward Compatibility

- Keep existing scripts in root directory during transition
- Mark as deprecated with migration notes
- Provide migration guide for each script

## Next Steps

1. Create folder structure
2. Implement core modules
3. Create example applications
4. Test with real data
5. Update documentation

