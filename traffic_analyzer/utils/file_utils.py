"""
File Utilities

File path and naming utilities.
"""

from pathlib import Path
from typing import Optional


def make_safe_filename(name: str, max_length: int = 100) -> str:
    """
    Convert string to safe filename.
    
    Args:
        name: Original filename
        max_length: Maximum length
        
    Returns:
        Safe filename
    """
    safe = "".join(c if c.isalnum() or c in (' ', '-', '_', '.') else '_' for c in name)
    return safe[:max_length]


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

