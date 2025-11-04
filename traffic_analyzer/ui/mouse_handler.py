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
                # Get frame dimensions from state for proper button scaling
                frame_width = getattr(state, 'width', None)
                frame_height = getattr(state, 'height', None)
                
                # Check button clicks first
                clicked_button = self.button_manager.get_clicked_button(x, y, frame_width, frame_height)
                if clicked_button:
                    if self.on_button_click:
                        self.on_button_click(clicked_button, state)
                    return
                
                # Handle ROI editing (check both state and roi_manager if available)
                poly_editing = False
                if hasattr(state, 'roi_manager'):
                    # Prefer roi_manager if available
                    poly_editing = getattr(state.roi_manager, 'poly_editing', False)
                    print(f"[DEBUG] Mouse click at ({x}, {y}), roi_manager.poly_editing = {poly_editing}")
                else:
                    # Fallback to state.roi
                    poly_editing = getattr(state.roi, 'poly_editing', False)
                    print(f"[DEBUG] Mouse click at ({x}, {y}), state.roi.poly_editing = {poly_editing}")
                
                if poly_editing:
                    print(f"[DEBUG] poly_editing is True, calling on_roi_click")
                    if self.on_roi_click:
                        self.on_roi_click((x, y), state)
                    else:
                        print(f"[DEBUG] on_roi_click callback is None!")
                        # Fallback: try to add point directly
                        if hasattr(state, 'roi_manager'):
                            state.roi_manager.add_point((x, y))
                            print(f"[DEBUG] Added point directly to roi_manager")
                        elif hasattr(state.roi, 'polygon_points'):
                            state.roi.polygon_points.append((x, y))
                            print(f"[DEBUG] Added point directly to state.roi")
                else:
                    print(f"[DEBUG] poly_editing is False, skipping ROI click")
                
                # Handle line editing
                edit_mode = False
                if hasattr(state, 'counting_line_obj'):
                    edit_mode = getattr(state.counting_line_obj, 'edit_mode', False)
                else:
                    edit_mode = getattr(state.counting_line, 'edit_mode', False)
                
                if edit_mode:
                    if hasattr(state, 'counting_line_obj'):
                        state.counting_line_obj.set_manual(y)
                    else:
                        state.counting_line.line_y = y
                        state.counting_line.mode = 'MANUAL'
                        state.counting_line.edit_mode = False
                    if self.on_line_click:
                        self.on_line_click(y, state)
        
        return callback

