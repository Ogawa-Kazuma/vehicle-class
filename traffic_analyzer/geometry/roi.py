"""
Region of Interest (ROI) Management

Handles polygon and rectangular ROI definition, masking, and point-in-polygon tests.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class ROIManager:
    """
    Manages Region of Interest (ROI) for detection filtering.
    
    Supports both polygon and rectangular ROI definitions.
    """
    
    def __init__(self):
        """Initialize ROI manager."""
        self.polygon_points: List[Tuple[int, int]] = []
        self.polygon_defined: bool = False
        self.poly_editing: bool = False
        self.polygon_mask: Optional[np.ndarray] = None
    
    def add_point(self, point: Tuple[int, int]):
        """Add a point to the polygon."""
        self.polygon_points.append(point)
    
    def remove_last_point(self):
        """Remove the last added point."""
        if self.polygon_points:
            self.polygon_points.pop()
    
    def finalize_polygon(self):
        """Finalize polygon definition (requires >= 3 points)."""
        if len(self.polygon_points) >= 3:
            self.polygon_defined = True
            self.poly_editing = False
    
    def reset(self):
        """Reset ROI state."""
        self.polygon_points.clear()
        self.polygon_defined = False
        self.poly_editing = False
        self.polygon_mask = None
    
    def rebuild_mask(self, frame_shape: Tuple[int, int, int]):
        """
        Rebuild polygon mask from current points.
        
        Args:
            frame_shape: (height, width, channels) tuple
        """
        self.polygon_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if len(self.polygon_points) >= 3:
            pts = np.array(self.polygon_points, dtype=np.int32)
            cv2.fillPoly(self.polygon_mask, [pts], 255)
    
    def point_in_polygon(self, point: Tuple[int, int]) -> bool:
        """
        Check if point is inside polygon.
        
        Args:
            point: (x, y) point to test
            
        Returns:
            True if point is inside polygon, False otherwise
        """
        if not self.polygon_defined or len(self.polygon_points) < 3:
            return True  # No ROI defined, accept all points
        
        pts = np.array(self.polygon_points, dtype=np.int32)
        result = cv2.pointPolygonTest(
            pts, 
            (float(point[0]), float(point[1])), 
            False
        )
        return result >= 0
    
    def apply_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply ROI mask to image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Masked image
        """
        if self.polygon_mask is None:
            return image
        
        if len(image.shape) == 3:
            # Color image
            return cv2.bitwise_and(image, image, mask=self.polygon_mask)
        else:
            # Grayscale
            return cv2.bitwise_and(image, self.polygon_mask)
    
    def get_polygon_mid_y(self) -> int:
        """
        Calculate midpoint Y coordinate of polygon.
        
        Returns:
            Y coordinate of polygon midpoint
        """
        if not self.polygon_points:
            return 250  # Default
        
        ys = [p[1] for p in self.polygon_points]
        return (min(ys) + max(ys)) // 2

