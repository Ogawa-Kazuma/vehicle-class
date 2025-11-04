"""
Vehicle Classification Module

Handles mapping between YOLO COCO classes and custom 6-class system.
Supports size-based classification and filtering.
"""

from typing import Dict, Tuple, Optional, Set
from collections import defaultdict


class VehicleClassifier:
    """
    Maps YOLO detections to custom vehicle classes.
    
    Supports:
    - COCO class mapping (car, motorcycle, bus, truck)
    - Size-based classification into 6 classes
    - Filtering by minimum size thresholds
    """
    
    # COCO class IDs
    COCO_CAR = 2
    COCO_MOTORCYCLE = 3
    COCO_BUS = 5
    COCO_TRUCK = 7
    
    # Default class names (Malay)
    DEFAULT_CLASSES = {
        0: 'Kelas 1',  # Motorcars, Taxis & Small MPVs
        1: 'Kelas 2',  # Small Vans, Big MPVs & Utilities
        2: 'Kelas 3',  # Lorries & Large Vans (2 axles)
        3: 'Kelas 4',  # Lorries with 3 axles & Construction vehicles
        4: 'Kelas 5',  # Bus
        5: 'Kelas 6',  # Motorcycles & Scooters
    }
    
    # Default size thresholds (min_width, min_height)
    DEFAULT_SIZE_THRESHOLDS = {
        'motorcycle': (5, 10),
        'car': (30, 50),
        'bus': (60, 100),
        'truck': (40, 90),
    }
    
    # Class mapping for 6-class system
    DEFAULT_SIX_CLASS_THRESHOLDS = {
        'Class 1': (60, 5),     # Motorcars, Taxis & Small MPVs
        'Class 2': (100, 120),  # Small Vans, Big MPVs & Utilities
        'Class 3': (120, 180),  # Lorries & Large Vans (2 axles)
        'Class 4': (180, 1000), # Lorries with 3 axles & Construction vehicles
        'Class 5': (120, 300),  # Bus
        'Class 6': (0, 60),     # Motorcycles & Scooters
    }
    
    def __init__(self, size_thresholds: Optional[Dict[str, Tuple[int, int]]] = None,
                 class_map: Optional[Dict[int, str]] = None,
                 use_six_class: bool = True):
        """
        Initialize classifier.
        
        Args:
            size_thresholds: Custom size thresholds dict
            class_map: Custom class mapping dict
            use_six_class: Use 6-class system (default) or simple COCO mapping
        """
        self.use_six_class = use_six_class
        self.class_map = class_map or self.DEFAULT_CLASSES
        self.size_thresholds = size_thresholds or (
            self.DEFAULT_SIX_CLASS_THRESHOLDS if use_six_class 
            else self.DEFAULT_SIZE_THRESHOLDS
        )
    
    def map_coco_to_custom(self, coco_class_id: int, width: int, height: int) -> Optional[str]:
        """
        Map COCO class ID to custom class based on size.
        
        Args:
            coco_class_id: YOLO COCO class ID (2=car, 3=motorcycle, 5=bus, 7=truck)
            width: Bounding box width
            height: Bounding box height
            
        Returns:
            Custom class name or None if not mapped
        """
        if not self.use_six_class:
            # Simple mapping: use COCO names directly
            simple_map = {
                self.COCO_CAR: 'car',
                self.COCO_MOTORCYCLE: 'motorcycle',
                self.COCO_BUS: 'bus',
                self.COCO_TRUCK: 'truck',
            }
            return simple_map.get(coco_class_id)
        
        # 6-class mapping with size thresholds
        if coco_class_id == self.COCO_CAR:
            if width >= self.size_thresholds['Class 1'][0] and width < self.size_thresholds['Class 2'][0]:
                return 'Class 1'
            elif width >= self.size_thresholds['Class 2'][0] and width < self.size_thresholds['Class 3'][0]:
                return 'Class 2'
            elif width >= self.size_thresholds['Class 3'][0] and width < self.size_thresholds['Class 4'][0]:
                return 'Class 3'
            elif width >= self.size_thresholds['Class 4'][0]:
                return 'Class 4'
        
        elif coco_class_id == self.COCO_TRUCK:
            if width >= self.size_thresholds['Class 3'][0] and width < self.size_thresholds['Class 4'][0]:
                return 'Class 3'
            elif width >= self.size_thresholds['Class 4'][0]:
                return 'Class 4'
        
        elif coco_class_id == self.COCO_BUS:
            if width >= self.size_thresholds['Class 5'][0]:
                return 'Class 5'
        
        elif coco_class_id == self.COCO_MOTORCYCLE:
            if width < self.size_thresholds['Class 6'][1]:
                return 'Class 6'
        
        return None
    
    def filter_by_size(self, class_name: str, width: int, height: int) -> bool:
        """
        Check if detection passes size threshold filter.
        
        Args:
            class_name: COCO class name (motorcycle, car, bus, truck)
            width: Bounding box width
            height: Bounding box height
            
        Returns:
            True if passes size filter, False otherwise
        """
        if class_name not in self.size_thresholds:
            return True  # No threshold defined, allow
        
        min_w, min_h = self.size_thresholds[class_name]
        return width >= min_w and height >= min_h
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get color for class visualization.
        
        Args:
            class_name: Class name
            
        Returns:
            RGB color tuple
        """
        color_map = {
            'Class 1': (255, 255, 255),  # White
            'Class 2': (0, 255, 0),      # Green
            'Class 3': (255, 0, 0),      # Red
            'Class 4': (200, 100, 0),    # Orange
            'Class 5': (0, 0, 255),      # Blue
            'Class 6': (255, 255, 0),    # Yellow
            'car': (255, 255, 255),
            'motorcycle': (255, 255, 0),
            'bus': (0, 0, 255),
            'truck': (255, 0, 0),
        }
        return color_map.get(class_name, (255, 255, 255))

