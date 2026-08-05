import numpy as np

def remove_green_noise(image_data: np.ndarray, strength: float = 1.0, method: str = "Average Neutral") -> np.ndarray:
    """
    Removes green chromatic noise (SCNR) from RGB image data.
    
    Parameters:
        image_data: float32 numpy array with shape (H, W, 3).
        strength: Blending strength from 0.0 (no change) to 1.0 (full effect).
        method: SCNR threshold method - "Average Neutral" or "Maximum Neutral".
    """
    # Guard against 2D grayscale data or single channel
    if image_data.ndim != 3 or image_data.shape[2] != 3:
        return image_data.copy()

    # Prevent out-of-range strength
    strength = np.clip(strength, 0.0, 1.0)
    if strength == 0.0:
        return image_data.copy()

    r = image_data[:, :, 0]
    g = image_data[:, :, 1]
    b = image_data[:, :, 2]

    # Calculate neutral green maximum based on other channels
    if method == "Maximum Neutral":
        g_max = np.maximum(r, b)
    else:  # "Average Neutral" default
        g_max = (r + b) / 2.0

    # Calculate excess green
    excess_green = np.maximum(0.0, g - g_max)

    # Subtract excess green scaled by strength
    g_corrected = g - (strength * excess_green)

    # Reconstruct RGB array
    output = np.zeros_like(image_data)
    output[:, :, 0] = r
    output[:, :, 1] = g_corrected
    output[:, :, 2] = b

    return output