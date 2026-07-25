from typing import Callable, Optional
import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter


def remove_stars(
    image: np.ndarray,
    detection_threshold: float = 2.5,
    star_radius: int = 5,
    inpaint_radius: int = 4,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Removes stars from linear/stretched astronomical images using multi-scale DoG

    detection and radial distance-weighted background infill.
    """
    if image.ndim not in (2, 3):
        raise ValueError("Star removal supports only 2-D or 3-D image arrays.")

    if progress_callback:
        progress_callback(10)

    source = image.astype(np.float32, copy=False)

    # 1. Normalize orientation to (H, W, C) for standard processing
    is_chw = False
    if source.ndim == 3 and source.shape[0] <= 4 and source.shape[0] < source.shape[2]:
        source = np.transpose(source, (1, 2, 0))
        is_chw = True

    # Extract luminance for detection
    if source.ndim == 2:
        luminance = source
    else:
        luminance = np.mean(source[:, :, :3], axis=2)

    if progress_callback:
        progress_callback(25)

    # 2. Difference of Gaussians (DoG) Star Detection
    sigma1 = max(0.8, star_radius * 0.3)
    sigma2 = max(3.0, star_radius * 1.5)

    blur1 = gaussian_filter(luminance, sigma=sigma1, mode="reflect")
    blur2 = gaussian_filter(luminance, sigma=sigma2, mode="reflect")
    dog = blur1 - blur2

    # Robust MAD noise estimation on positive DoG response
    pos_dog = dog[dog > 0]
    if len(pos_dog) == 0:
        return image.copy()

    med = np.median(pos_dog)
    mad = np.median(np.abs(pos_dog - med))
    sigma = mad / 0.6745 if mad > 0 else np.std(pos_dog)

    # 3. Create Star Mask
    mask = dog > (med + detection_threshold * sigma)

    # Expand mask based on inpaint_radius to cover star halos
    dilation_size = max(3, 2 * inpaint_radius + 1)
    footprint = np.ones((dilation_size, dilation_size), dtype=bool)
    mask = binary_dilation(mask, structure=footprint)

    if progress_callback:
        progress_callback(50)

    # 4. Distance-Transform Radial Infill (Preserves multi-channel integrity)
    # Compute nearest distance to unmasked pixels
    distances, indices = distance_transform_edt(
        mask, return_distances=True, return_indices=True
    )

    # Map masked pixels directly to their nearest valid background neighbor
    bg_fill = np.zeros_like(source)
    if source.ndim == 2:
        bg_fill = source[indices[0], indices[1]]
        # Smooth the filled regions slightly to blend seamless transitions
        bg_fill = cv2.GaussianBlur(bg_fill, (0, 0), sigmaX=2.0, sigmaY=2.0)
        result = np.where(mask, bg_fill, source)
    else:
        for c in range(source.shape[2]):
            channel_fill = source[:, :, c][indices[0], indices[1]]
            bg_fill[:, :, c] = cv2.GaussianBlur(
                channel_fill, (0, 0), sigmaX=2.0, sigmaY=2.0
            )

        mask_3d = mask[:, :, None]
        result = np.where(mask_3d, bg_fill, source)

    if progress_callback:
        progress_callback(85)

    # Restore original shape format if input was (C, H, W)
    if is_chw:
        result = np.transpose(result, (2, 0, 1))

    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        result = np.clip(result, limits.min, limits.max)

    if progress_callback:
        progress_callback(100)

    return result.astype(image.dtype, copy=False)