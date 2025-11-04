"""Detection modules for vehicle detection."""

from traffic_analyzer.detection.yolo_detector import YOLODetector
from traffic_analyzer.detection.motion_detector import MotionDetector
from traffic_analyzer.detection.hybrid_detector import HybridDetector

__all__ = ['YOLODetector', 'MotionDetector', 'HybridDetector']

