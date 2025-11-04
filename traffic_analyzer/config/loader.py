"""
Configuration Loader Module

Handles loading and saving configuration from JSON files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from traffic_analyzer.config.defaults import DEFAULT_CONFIG


class ConfigLoader:
    """
    Loads and saves application configuration.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory for config files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def get_config_path(self, video_path: str) -> Path:
        """
        Get config file path for video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Config file path
        """
        video_name = Path(video_path).stem
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in video_name)
        return self.config_dir / f"{safe_name}.json"
    
    def load(self, video_path: str) -> Dict[str, Any]:
        """
        Load configuration for video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Configuration dictionary
        """
        config_path = self.get_config_path(video_path)
        
        if not config_path.exists():
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Merge with defaults
            merged = DEFAULT_CONFIG.copy()
            merged.update(config)
            
            # Convert perspective points back to numpy array if present
            if 'perspective_source' in merged and merged['perspective_source']:
                merged['perspective_source'] = np.array(
                    merged['perspective_source'], 
                    dtype=np.float32
                )
            
            return merged
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}")
            return DEFAULT_CONFIG.copy()
    
    def save(self, video_path: str, config: Dict[str, Any]):
        """
        Save configuration for video.
        
        Args:
            video_path: Path to video file
            config: Configuration dictionary
        """
        config_path = self.get_config_path(video_path)
        
        # Convert numpy arrays to lists for JSON
        save_config = config.copy()
        if 'perspective_source' in save_config and save_config['perspective_source'] is not None:
            if isinstance(save_config['perspective_source'], np.ndarray):
                save_config['perspective_source'] = save_config['perspective_source'].tolist()
        
        try:
            with open(config_path, 'w') as f:
                json.dump(save_config, f, indent=2)
            print(f"[INFO] Saved config to {config_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")
    
    def save_roi(self, video_path: str, polygon_points: List[Tuple[int, int]]):
        """Save ROI polygon points."""
        config = self.load(video_path)
        config['roi_polygon'] = polygon_points
        self.save(video_path, config)
    
    def save_counting_line(self, video_path: str, line_y: int, mode: str = 'MANUAL'):
        """Save counting line position."""
        config = self.load(video_path)
        config['counting_line_y'] = line_y
        config['counting_line_mode'] = mode
        self.save(video_path, config)
    
    def save_perspective(self, video_path: str, source_points: List[Tuple[float, float]]):
        """Save perspective transform points."""
        config = self.load(video_path)
        config['perspective_source'] = source_points
        self.save(video_path, config)

