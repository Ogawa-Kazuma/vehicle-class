"""
YOLO Detector Module

Wrapper for Ultralytics YOLO models with consistent interface.
"""

import torch
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
import numpy as np


class YOLODetector:
    """
    YOLO-based vehicle detector.
    
    Wraps Ultralytics YOLO models with vehicle-specific filtering
    and class mapping.
    """
    
    # COCO class IDs for vehicles
    COCO_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    def __init__(self, model_path: str = 'yolov8s.pt', 
                 device: Optional[str] = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model (.pt file)
            device: Device to use ('cpu', '0', 'cuda:0', or None for auto)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-select device
        if device is None:
            self.device = '0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
    
    def detect(self, frame: np.ndarray, 
               classes: Optional[List[int]] = None) -> List[Dict]:
        """
        Detect vehicles in frame.
        
        Args:
            frame: Input frame (BGR format)
            classes: COCO class IDs to detect (default: [2,3,5,7])
            
        Returns:
            List of detection dicts with keys:
            - 'bbox': (x1, y1, x2, y2)
            - 'confidence': float
            - 'class_id': int (COCO class ID)
            - 'class_name': str (COCO class name)
            - 'center': (x, y) tuple
            - 'width': int
            - 'height': int
        """
        if classes is None:
            classes = self.COCO_CLASSES
        
        results = self.model(
            frame,
            classes=classes,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': cls_id,
                'class_name': cls_name,
                'center': center,
                'width': w,
                'height': h,
            })
        
        return detections
    
    def track(self, frame: np.ndarray,
              classes: Optional[List[int]] = None,
              persist: bool = True) -> List[Dict]:
        """
        Detect and track vehicles in frame.
        
        Args:
            frame: Input frame (BGR format)
            classes: COCO class IDs to detect
            persist: Whether to persist tracks across frames
            
        Returns:
            List of detection dicts with additional 'track_id' key
        """
        if classes is None:
            classes = self.COCO_CLASSES
        
        results = self.model.track(
            frame,
            classes=classes,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            tracker='bytetrack.yaml',
            persist=persist,
            verbose=False
        )[0]
        
        detections = []
        for box in results.boxes:
            if box.id is None:
                continue  # No track ID assigned
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            track_id = int(box.id[0].item())
            
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': cls_id,
                'class_name': cls_name,
                'center': center,
                'width': w,
                'height': h,
                'track_id': track_id,
            })
        
        return detections

