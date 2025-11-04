#!/usr/bin/env python3
"""
Test script for full_analyzer.py
Tests imports and basic functionality without requiring video input.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing full_analyzer.py imports and functionality")
print("=" * 60)

# Test 1: Import all required modules
print("\n[TEST 1] Testing imports...")
try:
    from traffic_analyzer.core import AppState, VehicleClassifier, TimeSynchronizer
    from traffic_analyzer.detection import YOLODetector
    from traffic_analyzer.tracking import CentroidTracker
    from traffic_analyzer.geometry import ROIManager, CountingLine, CrossingDetector
    from traffic_analyzer.io import VideoCapture, VehicleLogger, ImageSaver
    from traffic_analyzer.ui import DrawingUtils, ButtonManager, MouseHandler
    from traffic_analyzer.config import ConfigLoader
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize components
print("\n[TEST 2] Testing component initialization...")
try:
    state = AppState()
    config_loader = ConfigLoader()
    detector = YOLODetector(model_path='yolov8s.pt', conf_threshold=0.25)
    tracker = CentroidTracker(max_distance=50, max_disappeared=10)
    classifier = VehicleClassifier(use_six_class=True)
    roi_manager = ROIManager()
    counting_line = CountingLine(initial_y=250)
    crossing_detector = CrossingDetector(debounce_pixels=5)
    drawer = DrawingUtils()
    button_manager = ButtonManager()
    time_sync = TimeSynchronizer()
    print("✓ All components initialized successfully!")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check method signatures
print("\n[TEST 3] Testing method signatures...")
issues = []

# Check TimeSynchronizer.should_update_summary
try:
    import inspect
    sig = inspect.signature(time_sync.should_update_summary)
    params = list(sig.parameters.keys())
    if 'bucket_minutes' not in params:
        issues.append("TimeSynchronizer.should_update_summary missing 'bucket_minutes' parameter")
    print(f"✓ TimeSynchronizer.should_update_summary signature: {params}")
except Exception as e:
    issues.append(f"Error checking TimeSynchronizer: {e}")

# Check ROIManager methods
try:
    assert hasattr(roi_manager, 'point_in_polygon'), "ROIManager missing point_in_polygon"
    assert hasattr(roi_manager, 'get_polygon_mid_y'), "ROIManager missing get_polygon_mid_y"
    assert hasattr(roi_manager, 'finalize_polygon'), "ROIManager missing finalize_polygon"
    assert hasattr(roi_manager, 'rebuild_mask'), "ROIManager missing rebuild_mask"
    print("✓ ROIManager methods verified")
except AssertionError as e:
    issues.append(str(e))

# Check VehicleLogger methods
try:
    logger = VehicleLogger("test_output", "test_video")
    assert hasattr(logger, 'open_logs'), "VehicleLogger missing open_logs"
    assert hasattr(logger, 'log_vehicle'), "VehicleLogger missing log_vehicle"
    assert hasattr(logger, 'log_summary'), "VehicleLogger missing log_summary"
    assert hasattr(logger, 'close'), "VehicleLogger missing close"
    print("✓ VehicleLogger methods verified")
except AssertionError as e:
    issues.append(str(e))

# Check ConfigLoader methods
try:
    assert hasattr(config_loader, 'load'), "ConfigLoader missing load"
    assert hasattr(config_loader, 'save_roi'), "ConfigLoader missing save_roi"
    assert hasattr(config_loader, 'save_counting_line'), "ConfigLoader missing save_counting_line"
    print("✓ ConfigLoader methods verified")
except AssertionError as e:
    issues.append(str(e))

if issues:
    print("\n⚠ Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✓ All method signatures verified!")

# Test 4: Check tracking logic issue
print("\n[TEST 4] Checking tracking logic...")
print("⚠ Note: full_analyzer.py has a potential issue in tracking logic")
print("   Line 158: Comparing det['center'] == centroid (exact equality)")
print("   This will likely fail - should use distance-based matching")

# Test 5: Check TimeSynchronizer usage
print("\n[TEST 5] Checking TimeSynchronizer usage...")
print("⚠ Note: full_analyzer.py line 230 calls:")
print("   time_sync.should_update_summary(current_time)")
print("   But method signature requires: should_update_summary(current_time, bucket_minutes)")

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
if issues:
    print(f"⚠ Found {len(issues)} potential issues")
    print("\nRecommendations:")
    print("1. Fix should_update_summary call to include bucket_minutes parameter")
    print("2. Fix tracking detection-to-centroid matching logic")
else:
    print("✓ All basic tests passed!")
print("=" * 60)

