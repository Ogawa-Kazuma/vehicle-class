"""
Hybrid Detector Module

Combines motion detection and YOLO detection for improved accuracy.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from traffic_analyzer.detection.motion_detector import MotionDetector
from traffic_analyzer.detection.yolo_detector import YOLODetector


class HybridDetector:
    """
    Hybrid detector combining motion and YOLO.
    
    Uses motion detection to identify regions of interest,
    then applies YOLO only to those regions for efficiency.
    """
    
    def __init__(self, yolo_model_path: str = 'yolov8s.pt',
                 motion_history: int = 500,
                 motion_var_threshold: float = 25):
        """
        Initialize hybrid detector.
        
        Args:
            yolo_model_path: Path to YOLO model
            motion_history: Motion detector history
            motion_var_threshold: Motion detector variance threshold
        """
        self.motion_detector = MotionDetector(
            history=motion_history,
            var_threshold=motion_var_threshold
        )
        self.yolo_detector = YOLODetector(model_path=yolo_model_path)
    
    def detect(self, frame: np.ndarray,
               roi_mask: Optional[np.ndarray] = None,
               use_motion_filter: bool = True) -> List[Dict]:
        """
        Detect vehicles using hybrid approach.
        
        Args:
            frame: Input frame
            roi_mask: Optional ROI mask
            use_motion_filter: If True, only run YOLO on motion regions
            
        Returns:
            List of detection dicts (same format as YOLODetector.detect)
        """
        if use_motion_filter:
            # Get motion boxes
            motion_boxes = self.motion_detector.detect(frame, roi_mask)
            
            if not motion_boxes:
                return []  # No motion, skip YOLO
            
            # For now, run YOLO on full frame (can be optimized to crop regions)
            # TODO: Optimize to only run YOLO on motion regions
        
        # Run YOLO detection
        detections = self.yolo_detector.detect(frame)
        
        # Filter by ROI if provided
        if roi_mask is not None:
            filtered = []
            for det in detections:
                center = det['center']
                if self._point_in_mask(center, roi_mask):
                    filtered.append(det)
            detections = filtered
        
        return detections
    
    def _point_in_mask(self, point: Tuple[int, int], mask: np.ndarray) -> bool:
        """Check if point is in mask."""
        x, y = point
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            return mask[y, x] > 0
        return False

