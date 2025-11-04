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
    Manages UI buttons.
    """
    
    def __init__(self):
        """Initialize button manager."""
        self.buttons: Dict[str, Button] = {}
        self._setup_default_buttons()
    
    def _setup_default_buttons(self):
        """Setup default button set."""
        self.add_button("Start", 10, 10, color=(0, 200, 0))
        self.add_button("Exit", 120, 10, color=(0, 0, 200))
        self.add_button("Poly", 10, 60, color=(100, 100, 0))
        self.add_button("Line", 120, 60, color=(200, 100, 0))
        self.add_button("Capture", 10, 110, color=(60, 60, 60))
    
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
        Draw all buttons on frame.
        
        Args:
            frame: Frame to draw on
        """
        for button in self.buttons.values():
            # Draw button rectangle
            cv2.rectangle(frame, (button.x, button.y),
                         (button.x + button.width, button.y + button.height),
                         button.color, -1)
            
            # Draw button border
            cv2.rectangle(frame, (button.x, button.y),
                         (button.x + button.width, button.y + button.height),
                         (255, 255, 255), 2)
            
            # Draw button text
            text_size = cv2.getTextSize(button.name, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.6, 2)[0]
            text_x = button.x + (button.width - text_size[0]) // 2
            text_y = button.y + (button.height + text_size[1]) // 2
            cv2.putText(frame, button.name, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, button.text_color, 2)
    
    def is_clicked(self, button_name: str, x: int, y: int) -> bool:
        """
        Check if button was clicked.
        
        Args:
            button_name: Button name
            x: Click X coordinate
            y: Click Y coordinate
            
        Returns:
            True if clicked
        """
        if button_name not in self.buttons:
            return False
        
        btn = self.buttons[button_name]
        return (btn.x <= x <= btn.x + btn.width and
                btn.y <= y <= btn.y + btn.height)
    
    def get_clicked_button(self, x: int, y: int) -> Optional[str]:
        """
        Get name of clicked button.
        
        Args:
            x: Click X coordinate
            y: Click Y coordinate
            
        Returns:
            Button name or None
        """
        for name, btn in self.buttons.items():
            if self.is_clicked(name, x, y):
                return name
        return None

