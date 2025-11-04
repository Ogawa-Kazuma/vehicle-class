"""
Counting Line Management

Handles counting line definition and auto/manual positioning.
"""

from typing import Optional


class CountingLine:
    """
    Manages counting line for vehicle counting.
    
    The counting line is used to detect when vehicles cross
    and should be counted.
    """
    
    def __init__(self, initial_y: int = 250):
        """
        Initialize counting line.
        
        Args:
            initial_y: Initial Y coordinate
        """
        self.line_y = initial_y
        self.mode = 'AUTO'  # 'AUTO' or 'MANUAL'
        self.edit_mode = False
    
    def set_manual(self, y: int):
        """
        Set line position manually.
        
        Args:
            y: Y coordinate
        """
        self.line_y = y
        self.mode = 'MANUAL'
        self.edit_mode = False
    
    def set_auto(self, y: int):
        """
        Set line position automatically (from ROI midpoint).
        
        Args:
            y: Y coordinate from ROI
        """
        self.line_y = y
        self.mode = 'AUTO'
        self.edit_mode = False
    
    def reset(self):
        """Reset to default position."""
        self.line_y = 250
        self.mode = 'AUTO'
        self.edit_mode = False

