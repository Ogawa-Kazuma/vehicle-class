"""
Drawing Utilities Module

Provides functions for drawing bounding boxes, trails, ROI, and text on frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class DrawingUtils:
    """
    Drawing utilities for visualization.
    """
    
    def __init__(self):
        """Initialize drawing utilities."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
    
    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                 color: Tuple[int, int, int] = (0, 255, 0),
                 label: Optional[str] = None):
        """
        Draw bounding box.
        
        Args:
            frame: Frame to draw on
            bbox: (x1, y1, x2, y2)
            color: BGR color tuple
            label: Optional label text
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        
        if label:
            label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 5),
                         (x1 + label_size[0], label_y + 5), color, -1)
            cv2.putText(frame, label, (x1, label_y), self.font,
                       self.font_scale, (255, 255, 255), self.thickness)
    
    def draw_trail(self, frame: np.ndarray, trail: List[Tuple[int, int]],
                  color: Tuple[int, int, int] = (255, 0, 255)):
        """
        Draw tracking trail.
        
        Args:
            frame: Frame to draw on
            trail: List of (x, y) points
            color: BGR color tuple
        """
        if len(trail) < 2:
            return
        
        points = np.array(trail, dtype=np.int32)
        cv2.polylines(frame, [points], False, color, 2)
        
        # Draw last point as circle
        if trail:
            cv2.circle(frame, trail[-1], 5, color, -1)
    
    def draw_polygon(self, frame: np.ndarray, points: List[Tuple[int, int]],
                    color: Tuple[int, int, int] = (0, 255, 255),
                    filled: bool = False):
        """
        Draw polygon ROI.
        
        Args:
            frame: Frame to draw on
            points: List of polygon points
            color: BGR color tuple
            filled: Whether to fill polygon
        """
        if len(points) == 0:
            return
        
        # Draw connecting lines if we have 2+ points
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            
            if filled:
                cv2.fillPoly(frame, [pts], color)
            else:
                # Draw lines between consecutive points
                for i in range(len(points)):
                    if i == 0:
                        continue
                    cv2.line(frame, points[i-1], points[i], color, 2)
                # If polygon is closed (3+ points), draw last line
                if len(points) >= 3:
                    cv2.line(frame, points[-1], points[0], color, 2)
        
        # Always draw points (even for single point)
        point_color = (0, 255, 255)  # Cyan color for points
        point_radius = 8  # Larger radius for visibility
        for i, pt in enumerate(points):
            # Draw filled circle
            cv2.circle(frame, pt, point_radius, point_color, -1)
            # Draw border
            cv2.circle(frame, pt, point_radius, (255, 255, 255), 2)
            # Draw point number
            cv2.putText(frame, str(i+1), (pt[0]-5, pt[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_counting_line(self, frame: np.ndarray, line_y: int,
                          color: Tuple[int, int, int] = (255, 0, 0)):
        """
        Draw counting line.
        
        Args:
            frame: Frame to draw on
            line_y: Y coordinate of line
            color: BGR color tuple
        """
        h, w = frame.shape[:2]
        cv2.line(frame, (0, line_y), (w, line_y), color, 2)
        cv2.putText(frame, "Counting Line", (10, line_y - 5),
                   self.font, self.font_scale, color, self.thickness)
    
    def draw_counts(self, frame: np.ndarray, counts: Dict[str, int],
                   position: Tuple[int, int] = (10, 30)):
        """
        Draw vehicle counts on frame.
        
        Args:
            frame: Frame to draw on
            counts: Dictionary of class -> count
            position: (x, y) position for text
        """
        y_offset = 0
        for cls, count in counts.items():
            text = f"{cls}: {count}"
            cv2.putText(frame, text, (position[0], position[1] + y_offset),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
            y_offset += 25
    
    def draw_info(self, frame: np.ndarray, info: Dict[str, str],
                 position: Tuple[int, int] = (10, 10)):
        """
        Draw general info text.
        
        Args:
            frame: Frame to draw on
            info: Dictionary of label -> value
            position: (x, y) position for text
        """
        y_offset = 0
        for label, value in info.items():
            text = f"{label}: {value}"
            cv2.putText(frame, text, (position[0], position[1] + y_offset),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
            y_offset += 20

