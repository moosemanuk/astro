import numpy as np
from scipy.ndimage import median_filter


def remove_hot_pixels(
    image: np.ndarray,
    threshold: float = 5.0,
    progress_callback=None
) -> np.ndarray:
    """Detects and replaces single-pixel hot/cold impulse spikes using a 3x3 local median comparison.

    Args:
        image: 2D or 3D numpy array float32 in [0, 1] or raw ADU values.
        threshold: Sigma/multiplier threshold. Lower values catch dimmer hot pixels;
          higher values protect true star cores (default 5.0).
        progress_callback: Optional callback for UI progress reporting.

    Returns:
        Cleaned image numpy array.
    """
    cleaned = image.copy()

    if progress_callback:
        progress_callback(10)

    # Handle RGB (H, W, C) vs Mono (H, W)
    is_rgb = (image.ndim == 3 and image.shape[2] == 3)
    channels = 3 if is_rgb else 1

    for c in range(channels):
        channel_data = cleaned[:, :, c] if is_rgb else cleaned

        # 1. Compute 3x3 local median
        med = median_filter(channel_data, size=3)

        # 2. Compute local absolute deviation from median
        diff = np.abs(channel_data - med)

        # 3. Estimate local background dispersion (MAD - Median Absolute Deviation)
        mad = median_filter(diff, size=3) + 1e-6

        # 4. Identify outliers exceeding the threshold multiplier
        hot_mask = diff > (threshold * mad)

        # 5. Replace outlier pixels with their local median value
        if is_rgb:
            cleaned[:, :, c][hot_mask] = med[hot_mask]
        else:
            cleaned[hot_mask] = med[hot_mask]

        if progress_callback:
            progress_callback(10 + int(80 * (c + 1) / channels))

    if progress_callback:
        progress_callback(100)

    return cleaned