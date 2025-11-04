"""
Logging Module

Handles CSV logging for individual vehicles and summary statistics.
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path


class VehicleLogger:
    """
    Logs vehicle detection events to CSV files.
    
    Supports:
    - Individual vehicle logs (per detection)
    - Summary logs (per minute/15-minute buckets)
    """
    
    def __init__(self, output_dir: str, video_name: str):
        """
        Initialize logger.
        
        Args:
            output_dir: Output directory for logs
            video_name: Video filename (without extension)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_name = video_name
        self.log_file = None
        self.summary_log_file = None
        self.csv_writer = None
        self.summary_writer = None
        
        # Summary tracking
        self.current_bucket = None
        self.bucket_counts = defaultdict(int)
    
    def open_logs(self):
        """Open CSV log files."""
        safe_name = self._make_safe_filename(self.video_name)
        
        # Individual vehicle log
        log_path = self.output_dir / f"{safe_name}_log.csv"
        self.log_file = open(log_path, 'w', newline='', buffering=1)
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            'Timestamp', 'ID', 'Type', 'Confidence', 'VideoTime', 'Image'
        ])
        
        # Summary log
        summary_path = self.output_dir / f"{safe_name}_summary_log.csv"
        self.summary_log_file = open(summary_path, 'w', newline='', buffering=1)
        self.summary_writer = csv.writer(self.summary_log_file)
        self.summary_writer.writerow([
            'Time', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Total'
        ])
    
    def log_vehicle(self, timestamp: datetime, vehicle_id: int, 
                   vehicle_type: str, confidence: float,
                   video_time: str, image_path: str = ""):
        """
        Log individual vehicle detection.
        
        Args:
            timestamp: Detection timestamp
            vehicle_id: Track ID
            vehicle_type: Vehicle class
            confidence: Detection confidence
            video_time: Video timestamp string
            image_path: Path to saved image (optional)
        """
        if not self.csv_writer:
            self.open_logs()
        
        self.csv_writer.writerow([
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            vehicle_id,
            vehicle_type,
            f"{confidence:.2f}",
            video_time,
            image_path
        ])
        
        # Update bucket counts
        bucket_key = timestamp.strftime("%Y-%m-%d %H:%M")
        self.bucket_counts[bucket_key] += 1
    
    def log_summary(self, timestamp: datetime, class_counts: Dict[str, int]):
        """
        Log summary statistics.
        
        Args:
            timestamp: Bucket timestamp
            class_counts: Dictionary of class -> count
        """
        if not self.summary_writer:
            self.open_logs()
        
        bucket_key = timestamp.strftime("%Y-%m-%d %H:%M")
        
        # Only log if bucket changed
        if bucket_key != self.current_bucket:
            if self.current_bucket is not None:
                # Write previous bucket
                self._write_summary_row(self.current_bucket, self.bucket_counts)
            
            # Reset for new bucket
            self.current_bucket = bucket_key
            self.bucket_counts = defaultdict(int)
        
        # Update counts
        for cls, count in class_counts.items():
            self.bucket_counts[cls] += count
    
    def _write_summary_row(self, bucket_key: str, counts: Dict[str, int]):
        """Write a summary row."""
        total = sum(counts.values())
        self.summary_writer.writerow([
            bucket_key,
            counts.get('Class 1', 0),
            counts.get('Class 2', 0),
            counts.get('Class 3', 0),
            counts.get('Class 4', 0),
            counts.get('Class 5', 0),
            counts.get('Class 6', 0),
            total
        ])
    
    def flush(self):
        """Flush logs to disk."""
        if self.log_file:
            self.log_file.flush()
        if self.summary_log_file:
            self.summary_log_file.flush()
    
    def close(self):
        """Close log files."""
        # Write final summary bucket
        if self.current_bucket:
            self._write_summary_row(self.current_bucket, self.bucket_counts)
        
        if self.log_file:
            self.log_file.close()
        if self.summary_log_file:
            self.summary_log_file.close()
    
    @staticmethod
    def _make_safe_filename(name: str) -> str:
        """Convert filename to safe format."""
        safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        return safe[:100]  # Limit length

