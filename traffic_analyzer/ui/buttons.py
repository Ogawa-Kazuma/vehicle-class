"""
Button Management Module

Handles UI button definitions and drawing.
"""

import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Button:
    """Button definition."""
    name: str
    x: int
    y: int
    width: int = 100
    height: int = 40
    color: Tuple[int, int, int] = (100, 100, 100)
    text_color: Tuple[int, int, int] = (255, 255, 255)


class ButtonManager:
    """
    Manages UI buttons with proportional scaling.
    """
    
    # Reference resolution for scaling (Full HD)
    REFERENCE_WIDTH = 1920
    REFERENCE_HEIGHT = 1080
    
    def __init__(self):
        """Initialize button manager."""
        self.buttons: Dict[str, Button] = {}
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale = 1.0  # Unified scale factor (average of x and y)
        self._setup_default_buttons()
    
    def _setup_default_buttons(self):
        """Setup default button set."""
        # Button positions and sizes are relative to reference resolution
        self.add_button("Start", 10, 10, color=(0, 200, 0))
        self.add_button("Exit", 120, 10, color=(0, 0, 200))
        self.add_button("Poly", 10, 60, color=(100, 100, 0))
        self.add_button("Line", 120, 60, color=(200, 100, 0))
        self.add_button("Capture", 10, 110, color=(60, 60, 60))
    
    def update_scale(self, frame_width: int, frame_height: int):
        """
        Update scale factors based on frame dimensions.
        
        Args:
            frame_width: Current frame width
            frame_height: Current frame height
        """
        self.scale_x = frame_width / self.REFERENCE_WIDTH
        self.scale_y = frame_height / self.REFERENCE_HEIGHT
        # Use average scale for consistent button proportions
        self.scale = (self.scale_x + self.scale_y) / 2.0
    
    def add_button(self, name: str, x: int, y: int,
                  width: int = 100, height: int = 40,
                  color: Tuple[int, int, int] = (100, 100, 100)):
        """
        Add a button.
        
        Args:
            name: Button name
            x: X position
            y: Y position
            width: Button width
            height: Button height
            color: Button color (BGR)
        """
        self.buttons[name] = Button(name, x, y, width, height, color)
    
    def draw(self, frame):
        """
        Draw all buttons on frame with proportional scaling.
        
        Args:
            frame: Frame to draw on
        """
        # Update scale based on frame dimensions
        if frame is not None and len(frame.shape) >= 2:
            frame_height, frame_width = frame.shape[:2]
            self.update_scale(frame_width, frame_height)
        
        for button in self.buttons.values():
            # Scale button dimensions and position
            scaled_x = int(button.x * self.scale_x)
            scaled_y = int(button.y * self.scale_y)
            scaled_width = int(button.width * self.scale)
            scaled_height = int(button.height * self.scale)
            scaled_border = max(1, int(2 * self.scale))
            
            # Draw button rectangle
            cv2.rectangle(frame, (scaled_x, scaled_y),
                         (scaled_x + scaled_width, scaled_y + scaled_height),
                         button.color, -1)
            
            # Draw button border
            cv2.rectangle(frame, (scaled_x, scaled_y),
                         (scaled_x + scaled_width, scaled_y + scaled_height),
                         (255, 255, 255), scaled_border)
            
            # Scale text
            text_scale = 0.6 * self.scale
            text_thickness = max(1, int(2 * self.scale))
            
            # Draw button text
            text_size = cv2.getTextSize(button.name, cv2.FONT_HERSHEY_SIMPLEX,
                                       text_scale, text_thickness)[0]
            text_x = scaled_x + (scaled_width - text_size[0]) // 2
            text_y = scaled_y + (scaled_height + text_size[1]) // 2
            cv2.putText(frame, button.name, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, button.text_color, text_thickness)
    
    def is_clicked(self, button_name: str, x: int, y: int, frame_width: Optional[int] = None, frame_height: Optional[int] = None) -> bool:
        """
        Check if button was clicked.
        
        Args:
            button_name: Button name
            x: Click X coordinate
            y: Click Y coordinate
            frame_width: Frame width for scaling (optional, uses current scale if None)
            frame_height: Frame height for scaling (optional, uses current scale if None)
            
        Returns:
            True if clicked
        """
        if button_name not in self.buttons:
            return False
        
        # Update scale if frame dimensions provided
        if frame_width is not None and frame_height is not None:
            self.update_scale(frame_width, frame_height)
        
        btn = self.buttons[button_name]
        scaled_x = int(btn.x * self.scale_x)
        scaled_y = int(btn.y * self.scale_y)
        scaled_width = int(btn.width * self.scale)
        scaled_height = int(btn.height * self.scale)
        
        return (scaled_x <= x <= scaled_x + scaled_width and
                scaled_y <= y <= scaled_y + scaled_height)
    
    def get_clicked_button(self, x: int, y: int, frame_width: Optional[int] = None, frame_height: Optional[int] = None) -> Optional[str]:
        """
        Get name of clicked button.
        
        Args:
            x: Click X coordinate
            y: Click Y coordinate
            frame_width: Frame width for scaling (optional)
            frame_height: Frame height for scaling (optional)
            
        Returns:
            Button name or None
        """
        for name, btn in self.buttons.items():
            if self.is_clicked(name, x, y, frame_width, frame_height):
                return name
        return None

