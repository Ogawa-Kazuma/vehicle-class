"""
Image Saving Module

Handles saving cropped vehicle images and annotated frames.
"""

import cv2
import os
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import numpy as np


class ImageSaver:
    """
    Saves vehicle crops and annotated frames.
    """
    
    def __init__(self, output_dir: str, video_name: str):
        """
        Initialize image saver.
        
        Args:
            output_dir: Output directory
            video_name: Video filename (without extension)
        """
        self.output_dir = Path(output_dir) / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_name = video_name
        self.counter = 0
    
    def save_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                 vehicle_id: int, vehicle_type: str,
                 timestamp: Optional[datetime] = None) -> str:
        """
        Save cropped vehicle image.
        
        Args:
            frame: Full frame
            bbox: Bounding box (x1, y1, x2, y2)
            vehicle_id: Track ID
            vehicle_type: Vehicle class
            timestamp: Detection timestamp
            
        Returns:
            Path to saved image
        """
        x1, y1, x2, y2 = bbox
        
        # Crop with padding
        padding = 10
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        crop = frame[y1:y2, x1:x2]
        
        # Generate filename
        if timestamp:
            time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        else:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        safe_type = vehicle_type.replace(" ", "_")
        filename = f"{self.video_name}_{time_str}_{vehicle_id}_{safe_type}_{self.counter}.jpg"
        filepath = self.output_dir / filename
        
        cv2.imwrite(str(filepath), crop)
        self.counter += 1
        
        return str(filepath)
    
    def save_snapshot(self, frame: np.ndarray, timestamp: Optional[datetime] = None) -> str:
        """
        Save annotated frame snapshot.
        
        Args:
            frame: Annotated frame
            timestamp: Snapshot timestamp
            
        Returns:
            Path to saved snapshot
        """
        if timestamp:
            time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        else:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{self.video_name}_snapshot_{time_str}.jpg"
        filepath = self.output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        return str(filepath)

