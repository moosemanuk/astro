# processing/stretch.py
import numpy as np


def normalise_image(data: np.ndarray) -> np.ndarray:
    """Normalizes data safely to [0, 1] range."""
    d_min, d_max = np.min(data), np.max(data)
    if d_max > d_min:
        return (data - d_min) / (d_max - d_min)
    return np.zeros_like(data)


def apply_black_point(img: np.ndarray, shadow_tolerance: float = 2.5) -> np.ndarray:
    """Subtracts background floor based on image median and std dev."""
    med = np.median(img)
    std = np.std(img)
    black_point = max(0.0, med - (shadow_tolerance * std))
    
    img_bp = np.maximum(0.0, img - black_point)
    max_val = np.max(img_bp)
    if max_val > 0:
        img_bp = img_bp / max_val
    return img_bp


# --- STRETCH ALGORITHMS ---

def auto_midtone_stretch(
    data: np.ndarray,
    target_background: float = 0.25,
    black_point_percentile: float = 0.1,
    white_point_percentile: float = 99.9,
) -> np.ndarray:
    """Create a display stretch using robust levels and a midtone transfer curve.

    Robust levels prevent isolated saturated stars and hot pixels from hiding
    faint background structure by setting an excessively high white point.
    """
    source = np.asarray(data, dtype=np.float32)
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
        return normalise_image(source).astype(np.float32)

    img = np.clip((source - black_point) / (white_point - black_point), 0.0, 1.0)

    background = float(np.median(img))
    target = float(np.clip(target_background, 0.001, 0.999))

    if background <= 0.0 or background >= 1.0:
        return img

    # Solve the MTF equation for the parameter that maps `background` to `target`.
    midtone = (
        background * (target - 1.0)
        / (target * (2.0 * background - 1.0) - background)
    )
    midtone = float(np.clip(midtone, 0.0005, 0.9995))

    numerator = (midtone - 1.0) * img
    denominator = (2.0 * midtone - 1.0) * img - midtone
    denominator = np.where(denominator == 0.0, 1e-7, denominator)

    return np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)

def ghs_stretch(data: np.ndarray, stretch_factor: float = 0.15, symmetry_point: float = 0.05) -> np.ndarray:
    """Generalised Hyperbolic Stretch (GHS).

    - stretch_factor: [0..1] UI value -> maps to stretch parameter D [0.1..200.0]
    - symmetry_point: [0..1] UI value -> focus point x0
    """
    img = normalise_image(data)
    
    # Map [0..1] UI input to logarithmic stretch strength D [0.1 .. 200.0]
    D = 0.1 + (np.power(10.0, stretch_factor * 2.3) - 1.0) * 10.0
    x0 = np.clip(symmetry_point, 0.0001, 0.9999)

    # Arc-hyperbolic transformation centered at symmetry point x0
    num = np.arcsinh(D * (img - x0)) - np.arcsinh(-D * x0)
    den = np.arcsinh(D * (1.0 - x0)) - np.arcsinh(-D * x0)

    # Safe division
    stretched = np.where(den != 0, num / den, img)
    stretched = np.clip(stretched, 0.0, 1.0)

    return (stretched * (data.max() - data.min()) + data.min()).astype(data.dtype)


def arcsinh_stretch(data: np.ndarray, factor: float = 0.25, black_point: float = 0.0) -> np.ndarray:
    """Arcsinh stretch.

    - factor: [0..1] normalized UI value -> factor [0.1..100.0]
    - black_point: [0..1] normalized UI value -> [0.0..0.5] clip floor
    """
    img = normalise_image(data)
    s_factor = max(0.1, factor * 5000.0)
    bp = black_point * 0.5

    img_clipped = np.clip(img - bp, 0.0, 1.0)
    stretched = np.arcsinh(img_clipped * s_factor) / np.arcsinh(s_factor)
    return (stretched * (data.max() - data.min()) + data.min()).astype(data.dtype)


import numpy as np

def midtone_transfer_function(data: np.ndarray, midtone: float = 0.0, shadow_clip: float = 0.0) -> np.ndarray:
    """Midtone Transfer Function (MTF).

    - midtone: [0.0..1.0] UI slider value. 
               0.0 = Passthrough / Linear (m = 0.5)
               1.0 = Aggressive Stretch (m = 0.0005)
    - shadow_clip: [0.0..1.0] Black point clipping floor.
    """
    # 1. Standardize image range to [0.0, 1.0] float using your pipeline normalizer
    img = normalise_image(data).astype(np.float32)

    # 2. Apply Shadow Clip (Black Point)
    if shadow_clip > 0.0:
        bp = np.clip(shadow_clip, 0.0, 0.99)
        img = np.clip(img - bp, 0.0, 1.0)
        range_rem = 1.0 - bp
        if range_rem > 0:
            img /= range_rem

    # If midtone is 0, return the black-point adjusted linear image
    if midtone <= 0.0:
        return img

    # 3. Logarithmic mapping of UI midtone (0..1) to MTF parameter m (0.5..0.0005)
    t = np.clip(midtone, 0.0, 1.0)
    m = 0.5 * (0.001 ** t)

    # 4. Standard MTF Formula
    num = (m - 1.0) * img
    den = (2.0 * m - 1.0) * img - m
    
    # Avoid division by zero
    den = np.where(den == 0, 1e-7, den)
    
    # 5. Return normalized [0.0, 1.0] float array directly (NO re-scaling to raw min/max)
    return np.clip(num / den, 0.0, 1.0)


def log_stretch(data: np.ndarray, scaling_factor: float = 0.20) -> np.ndarray:
    """Logarithmic intensity stretch.

    - scaling_factor: [0..1] normalized UI value -> exponential log scale a [1..10000]
    """
    img = normalise_image(data)
    a = 1.0 + (np.power(10.0, scaling_factor * 4.0) - 1.0)

    stretched = np.log1p(a * img) / np.log1p(a)
    return (stretched * (data.max() - data.min()) + data.min()).astype(data.dtype)


def root_stretch(data: np.ndarray, root_power: float = 0.20) -> np.ndarray:
    """Root Stretch (Gamma / Power-Law).

    - root_power: [0..1] normalized UI value -> p [1.0..10.0]
    """
    img = normalise_image(data)
    p = 1.0 + root_power * 9.0

    stretched = np.power(img, 1.0 / p)
    return (stretched * (data.max() - data.min()) + data.min()).astype(data.dtype)


def exp_stretch(data: np.ndarray, exponent_slope: float = 0.15) -> np.ndarray:
    """Exponential Stretch.

    - exponent_slope: [0..1] normalized UI value -> b [0.1..20.0]
    """
    img = normalise_image(data)
    b = 0.1 + exponent_slope * 19.9

    stretched = (1.0 - np.exp(-b * img)) / (1.0 - np.exp(-b))
    return (stretched * (data.max() - data.min()) + data.min()).astype(data.dtype)
