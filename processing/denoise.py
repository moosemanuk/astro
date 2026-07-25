"""Detail-preserving denoising for monochrome and colour FITS image arrays."""

import numpy as np
from scipy.ndimage import gaussian_filter


def _denoise_channel(
    channel: np.ndarray,
    strength: float,
    smoothing_radius: float,
    detail_threshold: float,
) -> np.ndarray:
    """Suppress small-scale noise while retaining features above the threshold."""
    source = channel.astype(np.float32, copy=False)
    smooth = gaussian_filter(source, sigma=smoothing_radius, mode="reflect")
    residual = source - smooth

    # Estimate the noise robustly so isolated stars and hot pixels do not set
    # the denoise threshold.  The MAD-to-sigma conversion assumes Gaussian noise.
    median_residual = float(np.median(residual))
    noise_sigma = float(np.median(np.abs(residual - median_residual)) / 0.6745)
    if not np.isfinite(noise_sigma) or noise_sigma <= np.finfo(np.float32).eps:
        return source.copy()

    threshold = strength * detail_threshold * noise_sigma
    # Soft thresholding removes low-amplitude noise but leaves bright stellar
    # detail and nebula structure present in larger residuals.
    retained_detail = np.sign(residual) * np.maximum(np.abs(residual) - threshold, 0.0)
    return smooth + retained_detail


from typing import Callable, Optional


def denoise_image(
    image: np.ndarray,
    strength: float = 0.5,
    smoothing_radius: float = 1.2,
    detail_threshold: float = 2.0,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Denoise an image using Gaussian base smoothing and soft detail thresholding.

    Parameters
    ----------
    image:
        A 2-D monochrome array or a 3-D image with channels on the first or final axis.
    strength:
        Noise-reduction amount in the range 0 to 1.  Higher values remove more
        small-scale noise.
    smoothing_radius:
        Gaussian smoothing scale in pixels.  Larger values address coarser noise.
    detail_threshold:
        Detail protection in estimated noise-sigma units.  Higher values preserve
        more stars and fine structure, at the cost of leaving more noise behind.
    """
    if image.ndim not in (2, 3):
        raise ValueError("Denoising supports only 2-D or 3-D image arrays.")

    if progress_callback:
        progress_callback(5)

    strength = float(np.clip(strength, 0.0, 1.0))
    smoothing_radius = float(np.clip(smoothing_radius, 0.1, 10.0))
    detail_threshold = float(np.clip(detail_threshold, 0.1, 10.0))
    if strength == 0.0:
        if progress_callback:
            progress_callback(100)
        return image.copy()

    original_dtype = image.dtype
    if image.ndim == 2:
        result = _denoise_channel(image, strength, smoothing_radius, detail_threshold)
        if progress_callback:
            progress_callback(80)
    elif image.shape[2] <= 4:
        num_ch = image.shape[2]
        channels = []
        for index in range(num_ch):
            res_c = _denoise_channel(image[:, :, index], strength, smoothing_radius, detail_threshold)
            channels.append(res_c)
            if progress_callback:
                progress_callback(int(10 + 70 * (index + 1) / num_ch))
        result = np.stack(channels, axis=2)
    elif image.shape[0] <= 4:
        num_ch = image.shape[0]
        channels = []
        for index in range(num_ch):
            res_c = _denoise_channel(image[index, :, :], strength, smoothing_radius, detail_threshold)
            channels.append(res_c)
            if progress_callback:
                progress_callback(int(10 + 70 * (index + 1) / num_ch))
        result = np.stack(channels, axis=0)
    else:
        raise ValueError("3-D images must have a channel axis containing four or fewer channels.")

    if np.issubdtype(original_dtype, np.integer):
        limits = np.iinfo(original_dtype)
        result = np.clip(result, limits.min, limits.max)

    if progress_callback:
        progress_callback(80)

    return result.astype(original_dtype, copy=False)
