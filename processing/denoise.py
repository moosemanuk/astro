"""Multiscale Starlet (À Trous) Wavelet Denoiser for linear FITS image arrays."""

from typing import Callable, Optional
import numpy as np
from scipy.signal import fftconvolve


def _b3_spline_kernel_2d(step: int) -> np.ndarray:
    """Construct a 2D 5x5 B3-spline kernel with step-size hole spacing (À Trous algorithm)."""
    base_1d = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float32)
    
    size = 1 + 4 * step
    kernel_1d = np.zeros(size, dtype=np.float32)
    kernel_1d[::step] = base_1d
    
    return np.outer(kernel_1d, kernel_1d)


def _denoise_channel_starlet(
    channel: np.ndarray,
    strength: float = 1.0,
    scales: int = 4,
    thresholds: tuple = (3.0, 2.0, 1.0, 0.5),
) -> np.ndarray:
    """Denoise a 2D float32 channel using multiscale Starlet wavelet soft-thresholding."""
    source = channel.astype(np.float32, copy=False)
    current_layer = source.copy()
    
    wavelet_layers = []

    # --- PHASE 1: Wavelet Decomposition (À Trous Algorithm) ---
    for scale in range(scales):
        step = 2**scale
        kernel = _b3_spline_kernel_2d(step)
        
        # Convolve to obtain next smoothed scale
        next_layer = fftconvolve(current_layer, kernel, mode="same").astype(np.float32)
        
        # Isolate detail layer at current scale
        detail = current_layer - next_layer
        wavelet_layers.append(detail)
        
        current_layer = next_layer

    # 'current_layer' contains the coarse low-frequency residual background
    residual = current_layer

    # --- PHASE 2: Multiscale Thresholding ---
    denoised_layers = []
    
    for idx, detail in enumerate(wavelet_layers):
        thresh_multiplier = thresholds[idx] if idx < len(thresholds) else 0.5
        
        # Estimate noise std deviation (sigma) per scale using Median Absolute Deviation (MAD)
        med = float(np.median(detail))
        mad = float(np.median(np.abs(detail - med)))
        sigma = mad / 0.6745
        
        threshold = thresh_multiplier * sigma
        
        if threshold > np.finfo(np.float32).eps:
            # Soft-thresholding: smooth attenuation around boundary
            abs_detail = np.abs(detail)
            thresholded = np.sign(detail) * np.maximum(0.0, abs_detail - threshold)
            
            # Scale thresholding effect by user strength parameter
            if strength < 1.0:
                thresholded = (detail * (1.0 - strength)) + (thresholded * strength)
                
            denoised_layers.append(thresholded)
        else:
            denoised_layers.append(detail)

    # --- PHASE 3: Reconstruction ---
    reconstructed = np.sum(denoised_layers, axis=0) + residual
    
    # Clip back to valid dynamic range of source channel
    c_min = float(np.min(source))
    c_max = float(np.max(source))
    return np.clip(reconstructed, c_min, c_max)


def denoise_image(
    image: np.ndarray,
    strength: float = 1.0,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Denoise an image using Multiscale Starlet Wavelet thresholding.

    Parameters
    ----------
    image:
        A 2-D monochrome array or a 3-D image with channels on the first or final axis.
    strength:
        Noise-reduction blend amount in the range 0.0 (original) to 1.0 (full denoise).
    progress_callback:
        Optional callable accepting integer percentage (0 to 100).
    """
    if image.ndim not in (2, 3):
        raise ValueError("Denoising supports only 2-D or 3-D image arrays.")

    if progress_callback:
        progress_callback(5)

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        if progress_callback:
            progress_callback(100)
        return image.copy()

    original_dtype = image.dtype

    # Handle 2D Monochrome
    if image.ndim == 2:
        result = _denoise_channel_starlet(image, strength=strength)
        if progress_callback:
            progress_callback(80)

    # Handle 3D [H, W, C]
    elif image.shape[2] <= 4:
        num_ch = image.shape[2]
        channels = []
        for index in range(num_ch):
            res_c = _denoise_channel_starlet(image[:, :, index], strength=strength)
            channels.append(res_c)
            if progress_callback:
                progress_callback(int(10 + 70 * (index + 1) / num_ch))
        result = np.stack(channels, axis=2)

    # Handle 3D [C, H, W]
    elif image.shape[0] <= 4:
        num_ch = image.shape[0]
        channels = []
        for index in range(num_ch):
            res_c = _denoise_channel_starlet(image[index, :, :], strength=strength)
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
        progress_callback(100)

    return result.astype(original_dtype, copy=False)