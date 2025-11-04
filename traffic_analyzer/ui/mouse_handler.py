"""
Mouse Event Handler Module

Handles mouse callbacks for ROI definition, line positioning, and button clicks.
"""

import cv2
from typing import Callable, Optional, Tuple, Any
from traffic_analyzer.ui.buttons import ButtonManager


class MouseHandler:
    """
    Handles mouse events for interactive UI.
    """
    
    def __init__(self, button_manager: ButtonManager,
                 on_roi_click: Optional[Callable] = None,
                 on_line_click: Optional[Callable] = None,
                 on_button_click: Optional[Callable] = None):
        """
        Initialize mouse handler.
        
        Args:
            button_manager: Button manager instance
            on_roi_click: Callback for ROI polygon point clicks
            on_line_click: Callback for counting line positioning
            on_button_click: Callback for button clicks
        """
        self.button_manager = button_manager
        self.on_roi_click = on_roi_click
        self.on_line_click = on_line_click
        self.on_button_click = on_button_click
    
    def create_callback(self, window_name: str, state: Any) -> Callable:
        """
        Create OpenCV mouse callback function.
        
        Args:
            window_name: OpenCV window name
            state: Application state object (for access to ROI, line, etc.)
            
        Returns:
            Callback function for cv2.setMouseCallback
        """
        def callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Check button clicks first
                clicked_button = self.button_manager.get_clicked_button(x, y)
                if clicked_button:
                    if self.on_button_click:
                        self.on_button_click(clicked_button, state)
                    return
                
                # Handle ROI editing
                if state.roi.poly_editing:
                    state.roi.add_point((x, y))
                    if self.on_roi_click:
                        self.on_roi_click((x, y), state)
                
                # Handle line editing
                elif state.counting_line.edit_mode:
                    state.counting_line.set_manual(y)
                    if self.on_line_click:
                        self.on_line_click(y, state)
        
        return callback

