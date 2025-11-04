"""
Time Synchronization Module

Handles time alignment between video timestamps and real-world time.
"""

from datetime import datetime, timedelta
from typing import Optional


class TimeSynchronizer:
    """
    Synchronizes video time with real-world time.
    
    Allows setting a start date/time and calculating current timestamp
    based on video frame position.
    """
    
    def __init__(self, start_date: Optional[datetime] = None,
                 start_time: Optional[datetime] = None):
        """
        Initialize time synchronizer.
        
        Args:
            start_date: Start date (datetime object)
            start_time: Start time (datetime object combining date and time)
        """
        self.start_date = start_date
        self.start_time = start_time or datetime.now()
        self.current_timestamp = self.start_time
        self.previous_timestamp = self.start_time
    
    def set_start_time(self, date_str: str, time_str: str) -> bool:
        """
        Set start date and time from strings.
        
        Args:
            date_str: Date string in format "dd/mm/YYYY" or "YYYY-MM-DD"
            time_str: Time string in format "HH:MM:SS" or "HH:MM"
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse date
            if "/" in date_str:
                date_obj = datetime.strptime(date_str, "%d/%m/%Y").date()
            else:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            # Parse time
            fmt = "%H:%M:%S" if time_str.count(":") == 2 else "%H:%M"
            time_obj = datetime.strptime(time_str, fmt).time()
            
            # Combine
            self.start_time = datetime.combine(date_obj, time_obj)
            self.current_timestamp = self.start_time
            self.previous_timestamp = self.start_time
            
            return True
        except ValueError as e:
            print(f"[ERROR] Time parsing failed: {e}")
            return False
    
    def calculate_current_time(self, frame_number: int, fps: float) -> datetime:
        """
        Calculate current timestamp based on video frame position.
        
        Args:
            frame_number: Current frame number (0-indexed)
            fps: Video frames per second
            
        Returns:
            Current datetime object
        """
        if not self.start_time or fps <= 0:
            return datetime.now()
        
        time_elapsed_seconds = (frame_number - 1) / fps
        return self.start_time + timedelta(seconds=time_elapsed_seconds)
    
    def format_video_time(self, frame_number: int, fps: float) -> str:
        """
        Format video time as HH:MM:SS string.
        
        Args:
            frame_number: Current frame number
            fps: Video frames per second
            
        Returns:
            Formatted time string
        """
        time_elapsed_seconds = frame_number / fps
        hours, remainder = divmod(time_elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    def should_update_summary(self, current_time: datetime, 
                             bucket_minutes: int = 1) -> bool:
        """
        Check if summary log should be updated (bucket rollover).
        
        Args:
            current_time: Current timestamp
            bucket_minutes: Minutes per summary bucket
            
        Returns:
            True if bucket should roll over
        """
        if self.previous_timestamp is None:
            return False
        
        # Check if we've crossed a minute boundary
        prev_minute = (self.previous_timestamp.minute // bucket_minutes) * bucket_minutes
        curr_minute = (current_time.minute // bucket_minutes) * bucket_minutes
        
        return curr_minute != prev_minute

