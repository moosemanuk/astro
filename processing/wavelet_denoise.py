"""
Multiscale Starlet (À Trous) Wavelet Denoising for astronomical linear FITS data.
Uses B3-spline filtering to isolate spatial frequencies and threshold noise.
"""

from typing import Callable, Optional
import numpy as np
# from scipy.ndimage import filter2d
from scipy.ndimage import convolve
from scipy.ndimage import convolve1d


def _b3_spline_kernel(step: int) -> np.ndarray:
    """Constructs a 2D 5x5 B3-spline kernel with step-size hole spacing (À Trous algorithm)."""
    # 1D B3-spline base: [1/16, 1/4, 3/8, 1/4, 1/16]
    base_1d = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float32)
    size = 1 + 4 * step
    kernel_1d = np.zeros(size, dtype=np.float32)
    kernel_1d[::step] = base_1d
    
    # return np.outer(kernel_1d, kernel_1d)
    return kernel_1d


def _denoise_channel_starlet(
    channel: np.ndarray, 
    scales: int = 4, 
    thresholds: tuple = (3.0, 2.0, 1.0, 0.5),
    strength: float = 1.0
) -> np.ndarray:
    """Denoises a 2D single channel using Starlet wavelet thresholding."""
    source = channel.astype(np.float32, copy=False)
    current_layer = source.copy()
    
    wavelet_layers = []
    
    # --- PHASE 1: Wavelet Decomposition ---
    for scale in range(scales):
        step = 2**scale
        kernel = _b3_spline_kernel(step)
        
        # Convolve with expanding kernel (reflect boundary for astro edges)
        next_layer = convolve(current_layer, kernel, mode="reflect")
        next_layer = convolve(next_layer, kernel, axis=0, mode="reflect")
        
        # Detail layer = difference between successive smoothings
        detail = current_layer - next_layer
        wavelet_layers.append(detail)
        
        current_layer = next_layer

    # 'current_layer' now holds the coarse residual background
    residual = current_layer

    # --- PHASE 2: Scale-Aware Thresholding ---
    denoised_layers = []
    
    for idx, detail in enumerate(wavelet_layers):
        thresh_multiplier = thresholds[idx] if idx < len(thresholds) else 0.5
        
        # Robust noise estimation (MAD = Median Absolute Deviation)
        med = float(np.median(detail))
        mad = float(np.median(np.abs(detail - med)))
        sigma = mad / 0.6745
        
        threshold = thresh_multiplier * sigma
        
        if threshold > np.finfo(np.float32).eps:
            # Soft thresholding (preserves continuous transitions)
            abs_detail = np.abs(detail)
            thresholded = np.sign(detail) * np.maximum(0.0, abs_detail - threshold)
            
            # Blend based on global strength
            if strength < 1.0:
                thresholded = (detail * (1.0 - strength)) + (thresholded * strength)
                
            denoised_layers.append(thresholded)
        else:
            denoised_layers.append(detail)

    # --- PHASE 3: Reconstruction ---
    # Sum all thresholded detail layers + the coarse residual
    reconstructed = np.sum(denoised_layers, axis=0) + residual
    return reconstructed


def wavelet_denoise_image(
    image: np.ndarray,
    strength: float = 1.0,
    scales: int = 4,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """
    Multiscale Starlet Wavelet Denoiser for 2D or 3D astronomical arrays.
    """
    if image.ndim not in (2, 3):
        raise ValueError("Wavelet denoising supports only 2-D or 3-D image arrays.")

    if progress_callback:
        progress_callback(5)

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        if progress_callback:
            progress_callback(100)
        return image.copy()

    original_dtype = image.dtype

    # Default progressive threshold multipliers for scales 1..4 (3.0 sigma down to 0.5 sigma)
    thresh_tuple = (3.0, 2.0, 1.0, 0.5)

    if image.ndim == 2:
        result = _denoise_channel_starlet(image, scales, thresh_tuple, strength)
    elif image.shape[2] <= 4:
        num_ch = image.shape[2]
        channels = []
        for i in range(num_ch):
            res_c = _denoise_channel_starlet(image[:, :, i], scales, thresh_tuple, strength)
            channels.append(res_c)
            if progress_callback:
                progress_callback(int(10 + 80 * (i + 1) / num_ch))
        result = np.stack(channels, axis=2)
    elif image.shape[0] <= 4:
        num_ch = image.shape[0]
        channels = []
        for i in range(num_ch):
            res_c = _denoise_channel_starlet(image[i, :, :], scales, thresh_tuple, strength)
            channels.append(res_c)
            if progress_callback:
                progress_callback(int(10 + 80 * (i + 1) / num_ch))
        result = np.stack(channels, axis=0)
    else:
        raise ValueError("3-D images must have four or fewer channels.")

    if progress_callback:
        progress_callback(100)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        result = np.clip(result, info.min, info.max)
        result = np.round(result)
    return result.astype(original_dtype, copy=False)