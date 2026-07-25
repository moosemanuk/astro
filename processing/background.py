# processing/background.py
import logging
import numpy as np

logger = logging.getLogger("BackgroundExtraction")


def generate_grid_samples(
    image_shape: tuple[int, ...], num_points: int = 64, margin_ratio: float = 0.05
) -> list[tuple[int, int]]:
    """Generates an evenly spaced grid of background sample coordinates."""
    if len(image_shape) == 3:
        height, width = image_shape[0], image_shape[1]
    else:
        height, width = image_shape[:2]

    side = int(np.sqrt(num_points))

    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    x_coords = np.linspace(margin_x, width - margin_x, side, dtype=int)
    y_coords = np.linspace(margin_y, height - margin_y, side, dtype=int)

    grid = []
    for x in x_coords:
        for y in y_coords:
            grid.append((int(x), int(y)))
    return grid


def _extract_2d_channel(
    channel: np.ndarray,
    sample_coords: list[tuple[int, int]],
    degree: int,
    sample_radius: int,
    channel_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Internal helper to fit and extract background with outlier rejection."""
    ch_label = f" [Channel {channel_idx}]" if channel_idx is not None else ""
    orig_dtype = channel.dtype
    height, width = channel.shape

    # 1. Collect initial sample patch medians
    raw_x, raw_y, raw_b = [], [], []
    for x, y in sample_coords:
        x_min, x_max = max(0, x - sample_radius), min(width, x + sample_radius + 1)
        y_min, y_max = max(0, y - sample_radius), min(height, y + sample_radius + 1)

        patch = channel[y_min:y_max, x_min:x_max]
        if patch.size > 0:
            raw_x.append(x)
            raw_y.append(y)
            raw_b.append(np.median(patch))

    raw_b = np.array(raw_b, dtype=np.float64)

    # 2. Outlier Rejection: Filter out samples that land on nebulae or bright stars
    # Samples above (median + 1.2 * std) are assumed to be signal, not background sky
    b_med = np.median(raw_b)
    b_std = np.std(raw_b)
    cutoff = b_med + 1.2 * b_std

    valid_mask = raw_b <= cutoff
    # Ensure we keep at least 25% of points if threshold is too aggressive
    if np.sum(valid_mask) < len(raw_b) * 0.25:
        valid_mask = raw_b <= np.percentile(raw_b, 35)

    x_samples = np.array(raw_x)[valid_mask]
    y_samples = np.array(raw_y)[valid_mask]
    z_samples = raw_b[valid_mask]

    logger.info(
        f"{ch_label} Rejection: Retained {len(z_samples)} / {len(raw_b)} background points "
        f"(Discarded nebula/star samples > {cutoff:.4f})"
    )

    # 3. Convert coordinates to normalized [-1, 1] range
    x_norm = (2.0 * x_samples / (width - 1)) - 1.0
    y_norm = (2.0 * y_samples / (height - 1)) - 1.0

    # 4. Construct normalized 2D Vandermonde matrix and fit
    vander = np.polynomial.polynomial.polyvander2d(
        x_norm, y_norm, [degree, degree]
    )
    coefs, _, _, _ = np.linalg.lstsq(vander, z_samples, rcond=None)
    coefs_2d = coefs.reshape((degree + 1, degree + 1))

    # 5. Evaluate grid surface model
    # Note: polygrid2d returns shape (width, height), so transpose to match (height, width) image layout
    x_grid = np.linspace(-1.0, 1.0, width)
    y_grid = np.linspace(-1.0, 1.0, height)
    bg_model = np.polynomial.polynomial.polygrid2d(x_grid, y_grid, coefs_2d).T

    # 6. Perform background subtraction
    sky_offset = np.median(bg_model)
    corrected_channel = channel.astype(np.float64) - bg_model + sky_offset

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        corrected_channel = np.clip(corrected_channel, info.min, info.max)

    return corrected_channel.astype(orig_dtype), bg_model.astype(orig_dtype)


from typing import Callable, Optional


def extract_background_poly(
    image: np.ndarray,
    sample_coords: list[tuple[int, int]] | None = None,
    num_points: int = 64,
    degree: int = 2,
    sample_radius: int = 8,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits a 2D polynomial background model with signal outlier rejection."""
    if progress_callback:
        progress_callback(5)

    if sample_coords is None:
        sample_coords = generate_grid_samples(image.shape, num_points=num_points)

    if progress_callback:
        progress_callback(15)

    # Case 1: 2D Grayscale
    if image.ndim == 2:
        corr, bg = _extract_2d_channel(
            image, sample_coords, degree, sample_radius
        )
        if progress_callback:
            progress_callback(80)
        return corr, bg

    # Case 2: 3D Color / Multi-channel (H, W, C)
    if image.ndim == 3 and image.shape[2] <= 4:
        corrected_channels, bg_models = [], []
        num_ch = image.shape[2]
        for c in range(num_ch):
            corr, bg = _extract_2d_channel(
                image[:, :, c], sample_coords, degree, sample_radius, channel_idx=c
            )
            corrected_channels.append(corr)
            bg_models.append(bg)
            if progress_callback:
                progress_callback(int(15 + 65 * (c + 1) / num_ch))

        if progress_callback:
            progress_callback(80)
        return np.stack(corrected_channels, axis=2), np.stack(bg_models, axis=2)

    # Case 3: 3D FITS shape (C, H, W)
    if image.ndim == 3 and image.shape[0] <= 4:
        corrected_channels, bg_models = [], []
        num_ch = image.shape[0]
        for c in range(num_ch):
            corr, bg = _extract_2d_channel(
                image[c, :, :], sample_coords, degree, sample_radius, channel_idx=c
            )
            corrected_channels.append(corr)
            bg_models.append(bg)
            if progress_callback:
                progress_callback(int(15 + 65 * (c + 1) / num_ch))

        if progress_callback:
            progress_callback(80)
        return np.stack(corrected_channels, axis=0), np.stack(bg_models, axis=0)

    raise ValueError(f"Unsupported image shape for background extraction: {image.shape}")