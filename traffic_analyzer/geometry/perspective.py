"""
Perspective Transform for Speed Estimation

Handles bird's-eye view transformation for accurate speed calculation.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class PerspectiveTransformer:
    """
    Transforms points from camera view to bird's-eye view.
    
    Used for speed estimation by converting pixel movement
    to real-world distance.
    """
    
    def __init__(self, source_points: Optional[List[Tuple[float, float]]] = None,
                 target_width: int = 25, target_height: int = 250):
        """
        Initialize perspective transformer.
        
        Args:
            source_points: 4 points in camera view (TL, TR, BR, BL)
            target_width: Width of bird's-eye view
            target_height: Height of bird's-eye view
        """
        self.source_points = source_points
        self.target_width = target_width
        self.target_height = target_height
        
        # Default target rectangle
        self.target = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ], dtype=np.float32)
        
        self.transform_matrix = None
        if source_points and len(source_points) == 4:
            self._update_matrix()
    
    def _update_matrix(self):
        """Update transformation matrix from source points."""
        if not self.source_points or len(self.source_points) != 4:
            return
        
        source = np.array(self.source_points, dtype=np.float32)
        self.transform_matrix = cv2.getPerspectiveTransform(source, self.target)
    
    def set_source_points(self, points: List[Tuple[float, float]]):
        """
        Set source points and update transform.
        
        Args:
            points: 4 points in order: TL, TR, BR, BL
        """
        if len(points) != 4:
            raise ValueError("Must provide exactly 4 points")
        
        self.source_points = points
        self._update_matrix()
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from camera view to bird's-eye view.
        
        Args:
            points: Array of (x, y) points
            
        Returns:
            Transformed points
        """
        if self.transform_matrix is None:
            raise RuntimeError("Transform matrix not initialized. Set source points first.")
        
        if points.size == 0:
            return points
        
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped_points, self.transform_matrix)
        return transformed.reshape(-1, 2)
    
    def is_initialized(self) -> bool:
        """Check if transformer is ready to use."""
        return self.transform_matrix is not None

