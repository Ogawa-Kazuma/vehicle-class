"""
Motion Detection Module

Background subtraction-based motion detection using MOG2.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class MotionDetector:
    """
    Motion detector using background subtraction.
    
    Uses MOG2 algorithm to detect moving objects.
    """
    
    def __init__(self, history: int = 500, 
                 var_threshold: float = 25,
                 detect_shadows: bool = True):
        """
        Initialize motion detector.
        
        Args:
            history: Number of frames for background model
            var_threshold: Variance threshold for background
            detect_shadows: Whether to detect shadows
        """
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        # Morphological kernel for noise removal
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.min_area = 500  # Minimum contour area
    
    def detect(self, frame: np.ndarray, 
               roi_mask: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect moving objects in frame.
        
        Args:
            frame: Input frame (BGR format)
            roi_mask: Optional ROI mask to apply
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
        
        # Morphological operations to clean up
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract bounding boxes
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
        
        return boxes
    
    def get_mask(self, frame: np.ndarray,
                 roi_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get foreground mask.
        
        Args:
            frame: Input frame
            roi_mask: Optional ROI mask
            
        Returns:
            Foreground mask (binary image)
        """
        fg_mask = self.back_sub.apply(frame)
        
        if roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
        
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        return fg_mask
    
    def set_min_area(self, area: int):
        """Set minimum contour area threshold."""
        self.min_area = area

