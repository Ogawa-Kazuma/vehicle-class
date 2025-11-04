"""
Application State Management

Centralized state management for the traffic analyzer application.
Keeps track of processing state, ROI, counting line, and vehicle counts.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from datetime import datetime


@dataclass
class ROIState:
    """Region of Interest (ROI) state."""
    polygon_points: List[Tuple[int, int]] = field(default_factory=list)
    polygon_defined: bool = False
    poly_editing: bool = False
    polygon_mask: Optional[any] = None  # numpy array


@dataclass
class CountingLineState:
    """Counting line state."""
    line_y: int = 250
    mode: str = 'AUTO'  # 'AUTO' or 'MANUAL'
    edit_mode: bool = False


@dataclass
class VehicleCounts:
    """Vehicle counting statistics."""
    by_class: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_group: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    counted_ids: Set[int] = field(default_factory=set)
    per_minute: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class TrackingState:
    """Object tracking state."""
    tracking_trails: Dict[int, List[Tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    track_info: Dict[int, Dict] = field(default_factory=dict)
    last_crop: Optional[any] = None  # numpy array
    last_label: str = ""


@dataclass
class TimeState:
    """Time synchronization state."""
    start_date: Optional[datetime] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    capture_time: Optional[datetime] = None
    current_timestamp: Optional[datetime] = None
    previous_timestamp: Optional[datetime] = None


class AppState:
    """
    Main application state container.
    
    Consolidates all global state variables into a single, manageable object.
    """
    
    def __init__(self):
        # Processing control
        self.running: bool = True
        self.start_processing: bool = False
        self.force_exit: bool = False
        
        # ROI and geometry
        self.roi = ROIState()
        self.counting_line = CountingLineState()
        
        # Vehicle tracking and counting
        self.vehicle_counts = VehicleCounts()
        self.tracking = TrackingState()
        
        # Time management
        self.time = TimeState()
        
        # Directional counting
        self.count_direction: str = 'down'  # 'down', 'up', 'both'
        
        # Output directory
        self.output_dir: str = "vehicle_captures"
        
        # Current mouse position
        self.current_mouse: Tuple[int, int] = (0, 0)
        
        # Capture flag
        self.do_capture: bool = False
    
    def reset_counts(self):
        """Reset all counting statistics."""
        self.vehicle_counts.by_class.clear()
        self.vehicle_counts.by_group.clear()
        self.vehicle_counts.counted_ids.clear()
        self.vehicle_counts.per_minute.clear()
    
    def reset_tracking(self):
        """Reset tracking state."""
        self.tracking.tracking_trails.clear()
        self.tracking.track_info.clear()
        self.tracking.last_crop = None
        self.tracking.last_label = ""
    
    def reset_roi(self):
        """Reset ROI state."""
        self.roi.polygon_points.clear()
        self.roi.polygon_defined = False
        self.roi.poly_editing = False
        self.roi.polygon_mask = None
    
    def reset_all(self):
        """Reset all application state."""
        self.reset_counts()
        self.reset_tracking()
        self.reset_roi()
        self.start_processing = False
        self.counting_line.edit_mode = False
        self.counting_line.mode = 'AUTO'

