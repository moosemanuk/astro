import numpy as np


def normalise_image(data: np.ndarray) -> np.ndarray:
    """Normalizes data safely to [0.0, 1.0] float array."""
    d_min = float(np.min(data))
    d_max = float(np.max(data))
    if d_max > d_min:
        return (data.astype(np.float32) - d_min) / (d_max - d_min)
    return np.zeros(data.shape, dtype=np.float32)


def apply_black_point_protection(
    data: np.ndarray, 
    protection_factor: float = 0.0
) -> np.ndarray:
    """Applies a smooth protection curve to low values (shadows/black point).
    
    - protection_factor = 0.0: No protection (stretches everything linearly near 0).
    - protection_factor > 0.0: Progressively dampens low values so sky background
      remains dark while midtones expand.
    """
    p = float(np.clip(protection_factor, 0.0, 1.0))
    if p == 0.0:
        return data

    # Power exponent scaling (1.0 to 4.0) to suppress low values
    power = 1.0 + (p * 3.0)
    return np.power(data, power).astype(np.float32)


def auto_midtone_stretch(
    data: np.ndarray,
    target_background: float = 0.25,
    black_point_percentile: float = 0.1,
    white_point_percentile: float = 99.9,
    protect_black_point: float = 1.0,
) -> np.ndarray:
    """Statistical Midtone Transfer Function (MTF) with optional shadow protection."""
    source = normalise_image(data)
    
    # Protect shadow region if enabled
    if protect_black_point > 0.0:
        source = apply_black_point_protection(source, protect_black_point)

    finite_values = source[np.isfinite(source)]
    if finite_values.size == 0:
        return np.zeros_like(source, dtype=np.float32)

    black_percentile = float(np.clip(black_point_percentile, 0.0, 99.0))
    white_percentile = float(
        np.clip(white_point_percentile, black_percentile + 0.1, 100.0)
    )
    black_point, white_point = np.percentile(
        finite_values, [black_percentile, white_percentile]
    )
    if not np.isfinite(black_point) or not np.isfinite(white_point) or white_point <= black_point:
        return source

    # Clip to percentile black/white bounds
    img = np.clip((source - black_point) / (white_point - black_point), 0.0, 1.0)

    bg_median = float(np.median(img))
    target = float(np.clip(target_background, 0.001, 0.999))

    if bg_median <= 0.0 or bg_median >= 1.0:
        return img

    midtone = (bg_median * (target - 1.0)) / (target * (2.0 * bg_median - 1.0) - bg_median)
    midtone = float(np.clip(midtone, 0.0001, 0.9999))

    num = (midtone - 1.0) * img
    den = (2.0 * midtone - 1.0) * img - midtone
    den = np.where(den == 0.0, 1e-7, den)

    return np.clip(num / den, 0.0, 1.0).astype(np.float32)


def midtone_transfer_function(
    data: np.ndarray, 
    target_median: float = 0.25, 
    shadow_clip: float = 0.001,
    protect_black_point: float = 0.0,
) -> np.ndarray:
    """Statistical MTF stretch taking a target median value directly (0.0 to 1.0)."""
    bp_pct = shadow_clip * 100.0
    return auto_midtone_stretch(
        data, 
        target_background=target_median, 
        black_point_percentile=bp_pct, 
        white_point_percentile=99.95,
        protect_black_point=protect_black_point
    )


def ghs_stretch(
    data: np.ndarray, 
    stretch_factor: float = 0.5, 
    symmetry_point: float = 0.005,
    local_intensity: float = 0.0,
    protect_black_point: float = 0.0,
) -> np.ndarray:
    """Generalised Hyperbolic Stretch (GHS) with shadow protection."""
    img = normalise_image(data)
    
    if protect_black_point > 0.0:
        img = apply_black_point_protection(img, protect_black_point)

    if stretch_factor <= 0.0:
        return img

    x = float(np.clip(stretch_factor, 0.0, 1.0))
    if x <= 0.0:
        return np.zeros_like(img, dtype=np.float32)
    warped_x = np.power(x, 0.4)

    bg_med = float(np.median(img))

    if symmetry_point is None or symmetry_point <= 0.0:
        x0 = bg_med
    else:
        x0 = float(symmetry_point)

    x0 = np.clip(x0, 0.00001, 0.9999)
    D = np.power(10.0, warped_x * 6) - 1.0
    b = float(np.clip(local_intensity, 0.0, 15.0))

    if b > 0.0:
        p = b * D * (img - x0)
        p0 = -b * D * x0
        p1 = b * D * (1.0 - x0)
        
        num = np.arcsinh(p) - np.arcsinh(p0)
        den = np.arcsinh(p1) - np.arcsinh(p0)
    else:
        num = np.arcsinh(D * (img - x0)) - np.arcsinh(-D * x0)
        den = np.arcsinh(D * (1.0 - x0)) - np.arcsinh(-D * x0)

    if den == 0:
        return img

    return np.clip(num / den, 0.0, 1.0).astype(np.float32)


def arcsinh_stretch(
    data: np.ndarray, 
    factor: float = 0.25, 
    black_point: float = 0.0,
    protect_black_point: float = 0.0,
) -> np.ndarray:
    """Arcsinh stretch with shadow protection."""
    img = normalise_image(data)
    
    if protect_black_point > 0.0:
        img = apply_black_point_protection(img, protect_black_point)

    s_factor = max(0.1, factor * 1000.0)
    bp = black_point * 0.1

    img_clipped = np.clip(img - bp, 0.0, 1.0)
    stretched = np.arcsinh(img_clipped * s_factor) / np.arcsinh(s_factor)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def log_stretch(
    data: np.ndarray, 
    scaling_factor: float = 0.20,
    protect_black_point: float = 0.0,
) -> np.ndarray:
    """Logarithmic intensity stretch with shadow protection."""
    img = normalise_image(data)
    
    if protect_black_point > 0.0:
        img = apply_black_point_protection(img, protect_black_point)

    a = 1.0 + (np.power(10.0, scaling_factor * 4.0) - 1.0)
    stretched = np.log1p(a * img) / np.log1p(a)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def root_stretch(
    data: np.ndarray, 
    root_power: float = 0.20,
    protect_black_point: float = 0.0,
) -> np.ndarray:
    """Power-Law/Root Stretch with shadow protection."""
    img = normalise_image(data)
    
    if protect_black_point > 0.0:
        img = apply_black_point_protection(img, protect_black_point)

    p = 1.0 + root_power * 19.0
    stretched = np.power(img, 1.0 / p)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def exp_stretch(
    data: np.ndarray, 
    exponent_slope: float = 0.15,
    protect_black_point: float = 0.0,
) -> np.ndarray:
    """Exponential Stretch with shadow protection."""
    img = normalise_image(data)
    
    if protect_black_point > 0.0:
        img = apply_black_point_protection(img, protect_black_point)

    b = 0.1 + exponent_slope * 29.9
    stretched = (1.0 - np.exp(-b * img)) / (1.0 - np.exp(-b))
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)