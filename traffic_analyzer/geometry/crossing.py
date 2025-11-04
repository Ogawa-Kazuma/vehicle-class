"""
Line Crossing Detection

Detects when tracked objects cross the counting line.
Supports directional counting (up, down, both).
"""

from typing import Optional


class CrossingDetector:
    """
    Detects when objects cross the counting line.
    
    Supports directional filtering and debounce logic.
    """
    
    def __init__(self, debounce_pixels: int = 5):
        """
        Initialize crossing detector.
        
        Args:
            debounce_pixels: Minimum pixels beyond line before counting
        """
        self.debounce_pixels = debounce_pixels
    
    def crossed_line(self, prev_y: Optional[int], curr_y: Optional[int], 
                    line_y: int, direction: str = 'down') -> bool:
        """
        Check if object crossed the counting line.
        
        Args:
            prev_y: Previous Y coordinate (bottom of bbox)
            curr_y: Current Y coordinate (bottom of bbox)
            line_y: Counting line Y coordinate
            direction: 'down', 'up', or 'both'
            
        Returns:
            True if crossed in specified direction
        """
        if prev_y is None or curr_y is None or prev_y == curr_y:
            return False
        
        # Check if actually crossed the line
        prev_side = prev_y - line_y
        curr_side = curr_y - line_y
        
        crossed = (
            (prev_side * curr_side) < 0 or  # Opposite sides
            (prev_side == 0 and curr_side != 0) or
            (curr_side == 0 and prev_side != 0)
        )
        
        if not crossed:
            return False
        
        # Determine crossing direction
        if curr_y > prev_y:
            # Moving down
            if direction in ['down', 'both']:
                # Check debounce
                if curr_y > line_y and (curr_y - line_y) >= self.debounce_pixels:
                    return True
        elif curr_y < prev_y:
            # Moving up
            if direction in ['up', 'both']:
                # Check debounce
                if curr_y < line_y and (line_y - curr_y) >= self.debounce_pixels:
                    return True
        
        return False
    
    def crossed_line_direction(self, prev_y: Optional[int], curr_y: Optional[int],
                               line_y: int) -> int:
        """
        Detect crossing direction.
        
        Args:
            prev_y: Previous Y coordinate
            curr_y: Current Y coordinate
            line_y: Counting line Y coordinate
            
        Returns:
            +1 for downward crossing, -1 for upward, 0 for no crossing
        """
        if prev_y is None or curr_y is None or prev_y == curr_y:
            return 0
        
        prev_side = prev_y - line_y
        curr_side = curr_y - line_y
        
        crossed = (
            (prev_side * curr_side) < 0 or
            (prev_side == 0 and curr_side != 0) or
            (curr_side == 0 and prev_side != 0)
        )
        
        if not crossed:
            return 0
        
        return +1 if curr_y > prev_y else -1

