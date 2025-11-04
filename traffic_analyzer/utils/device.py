"""
Device Selection Utilities

Handles GPU/CPU selection and device availability reporting.
"""

import torch
import subprocess
from typing import Optional


def select_device(device: Optional[str] = None) -> str:
    """
    Select computation device.
    
    Args:
        device: Device string ('cpu', '0', 'cuda:0') or None for auto-select
        
    Returns:
        Device string
    """
    if device:
        return device
    
    if torch.cuda.is_available():
        return '0'  # Use first GPU
    else:
        return 'cpu'


def gpu_availability_report() -> dict:
    """
    Get detailed GPU availability information.
    
    Returns:
        Dictionary with GPU info
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'nvenc_available': False,
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        
        # Check for NVENC
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                capture_output=True,
                text=True,
                timeout=5
            )
            info['nvenc_available'] = 'h264_nvenc' in result.stdout
        except:
            pass
    
    return info

