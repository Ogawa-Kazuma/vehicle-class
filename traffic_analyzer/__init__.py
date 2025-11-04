"""
Traffic Analyzer - Modular Vehicle Detection and Tracking System

A refactored, modular codebase for traffic analysis with:
- Multiple detection methods (YOLO, motion-based, hybrid)
- Robust tracking (CentroidTracker, ByteTrack)
- Flexible ROI management (polygon, rectangular)
- Comprehensive logging and MQTT integration
"""

__version__ = "2.0.0"
__author__ = "Traffic Analyzer Team"

from traffic_analyzer.core.state import AppState
from traffic_analyzer.core.vehicle_classifier import VehicleClassifier

__all__ = ['AppState', 'VehicleClassifier']

