"""Unsharp-mask sharpening with optional stellar-detail targeting."""

import numpy as np
from scipy.ndimage import gaussian_filter


def _sharpen_channel(
    channel: np.ndarray,
    strength: float,
    radius: float,
    target: str,
    stellar_threshold: float,
) -> np.ndarray:
    source = channel.astype(np.float32, copy=False)
    blurred = gaussian_filter(source, sigma=radius, mode="reflect")
    detail = source - blurred

    if target != "both":
        # Estimate fine-scale noise robustly, then classify only pronounced
        # positive high-frequency detail as stellar.  This is intentionally
        # conservative so diffuse nebula is not mistaken for a star field.
        noise = float(np.median(np.abs(detail - np.median(detail))) / 0.6745)
        noise = max(noise, np.finfo(np.float32).eps)
        transition = max(noise * 0.5, np.finfo(np.float32).eps)
        stellar_mask = np.clip(
            (detail - stellar_threshold * noise) / transition, 0.0, 1.0
        )
        if target == "stellar":
            detail *= stellar_mask
        else:
            detail *= 1.0 - stellar_mask

    return source + strength * detail


from typing import Callable, Optional


def sharpen_image(
    image: np.ndarray,
    strength: float = 0.75,
    radius: float = 1.2,
    target: str = "both",
    stellar_threshold: float = 3.0,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Sharpen a 2-D or RGB image with an unsharp mask.

    ``target`` may be ``"both"``, ``"stellar"``, or ``"non_stellar"``.
    The stellar threshold is measured in estimated fine-detail noise sigma.
    """
    if image.ndim not in (2, 3):
        raise ValueError("Sharpening supports only 2-D or 3-D image arrays.")
    if target not in {"both", "stellar", "non_stellar"}:
        raise ValueError("target must be 'both', 'stellar', or 'non_stellar'.")

    if progress_callback:
        progress_callback(5)

    strength = float(np.clip(strength, 0.0, 5.0))
    radius = float(np.clip(radius, 0.1, 20.0))
    stellar_threshold = float(np.clip(stellar_threshold, 0.5, 20.0))
    dtype = image.dtype

    if image.ndim == 2:
        result = _sharpen_channel(image, strength, radius, target, stellar_threshold)
        if progress_callback:
            progress_callback(80)
    elif image.shape[2] <= 4:
        num_ch = image.shape[2]
        channels = []
        for c in range(num_ch):
            res_c = _sharpen_channel(image[:, :, c], strength, radius, target, stellar_threshold)
            channels.append(res_c)
            if progress_callback:
                progress_callback(int(10 + 70 * (c + 1) / num_ch))
        result = np.stack(channels, axis=2)
    elif image.shape[0] <= 4:
        num_ch = image.shape[0]
        channels = []
        for c in range(num_ch):
            res_c = _sharpen_channel(image[c, :, :], strength, radius, target, stellar_threshold)
            channels.append(res_c)
            if progress_callback:
                progress_callback(int(10 + 70 * (c + 1) / num_ch))
        result = np.stack(channels, axis=0)
    else:
        raise ValueError("3-D images must have a channel axis containing four or fewer channels.")

    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        result = np.clip(result, limits.min, limits.max)

    if progress_callback:
        progress_callback(80)

    return result.astype(dtype, copy=False)
