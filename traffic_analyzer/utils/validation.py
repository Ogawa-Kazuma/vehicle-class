"""
Validation Utilities

Input validation functions.
"""

import os
from pathlib import Path


def validate_video_path(video_path: str) -> bool:
    """
    Validate video file path.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, False otherwise
    """
    if not video_path:
        return False
    
    path = Path(video_path)
    if not path.exists():
        return False
    
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    return path.suffix.lower() in valid_extensions

