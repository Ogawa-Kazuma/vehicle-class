"""
Video I/O Module

Handles video capture and streaming functionality.
"""

import cv2
import subprocess
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path


class VideoCapture:
    """
    Video capture wrapper with enhanced functionality.
    """
    
    def __init__(self, source: str):
        """
        Initialize video capture.
        
        Args:
            source: Video file path or camera index
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame."""
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def get_frame_number(self) -> int:
        """Get current frame number."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def set_frame(self, frame_number: int):
        """Set frame position."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoStreamer:
    """
    FFmpeg-based video streaming.
    
    Supports RTSP/RTMP streaming with optional NVENC acceleration.
    """
    
    def __init__(self, output_url: str, width: int, height: int, 
                 fps: float = 30.0, use_nvenc: bool = False):
        """
        Initialize video streamer.
        
        Args:
            output_url: RTSP/RTMP output URL
            width: Frame width
            height: Frame height
            fps: Frame rate
            use_nvenc: Use NVENC hardware encoding
        """
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.use_nvenc = use_nvenc
        self.process: Optional[subprocess.Popen] = None
    
    def start(self):
        """Start streaming process."""
        if self.process:
            return  # Already started
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-s', f'{self.width}x{self.height}', '-pix_fmt', 'bgr24',
               '-r', str(self.fps), '-i', '-', '-an']
        
        if self.use_nvenc:
            cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'fast'])
        else:
            cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast'])
        
        cmd.extend(['-f', 'rtsp', self.output_url])
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    def write(self, frame: np.ndarray):
        """
        Write frame to stream.
        
        Args:
            frame: Frame to write (BGR format)
        """
        if not self.process:
            self.start()
        
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
            except BrokenPipeError:
                pass  # Stream closed
    
    def close(self):
        """Close streaming process."""
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            self.process = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

