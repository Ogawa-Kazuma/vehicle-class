"""
ByteTrack Wrapper

Wrapper for supervision ByteTrack integration.
Provides consistent interface with CentroidTracker.
"""

from typing import List, Tuple, Dict, Optional
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    sv = None


class ByteTrackWrapper:
    """
    Wrapper for ByteTrack tracker from supervision library.
    
    Provides a simpler interface and falls back gracefully
    if supervision is not available.
    """
    
    def __init__(self, frame_rate: float = 30.0):
        """
        Initialize ByteTrack wrapper.
        
        Args:
            frame_rate: Video frame rate for tracker
        """
        if not SUPERVISION_AVAILABLE:
            raise ImportError(
                "supervision library required for ByteTrack. "
                "Install with: pip install supervision"
            )
        
        self.tracker = sv.ByteTrack(frame_rate=min(frame_rate, 30.0))
        self.frame_rate = frame_rate
    
    def update(self, detections: any) -> any:
        """
        Update tracker with new detections.
        
        Args:
            detections: supervision Detections object
            
        Returns:
            Updated detections with tracker_id assigned
        """
        if detections is None or len(detections) == 0:
            return detections
        
        return self.tracker.update_with_detections(detections)
    
    def get_tracked_ids(self, detections: any) -> List[int]:
        """
        Extract tracked object IDs from detections.
        
        Args:
            detections: supervision Detections object with tracker_id
            
        Returns:
            List of tracked object IDs
        """
        if detections is None or detections.tracker_id is None:
            return []
        
        return [int(tid) for tid in detections.tracker_id if tid != -1]

