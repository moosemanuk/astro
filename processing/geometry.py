# processing/geometry.py
import numpy as np

def flip_horizontal(data: np.ndarray) -> np.ndarray:
    """Flips the image array horizontally (left to right)."""
    if data is None:
        return None
    # Slices along the width axis (axis 1)
    return np.flip(data, axis=1)

def flip_vertical(data: np.ndarray) -> np.ndarray:
    """Flips the image array vertically (top to bottom)."""
    if data is None:
        return None
    # Slices along the height axis (axis 0)
    return np.flip(data, axis=0)