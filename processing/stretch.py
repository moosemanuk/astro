import numpy as np


def normalise_image(data: np.ndarray) -> np.ndarray:
    """Normalizes data safely to [0.0, 1.0] float array."""
    d_min = float(np.min(data))
    d_max = float(np.max(data))
    if d_max > d_min:
        return (data.astype(np.float32) - d_min) / (d_max - d_min)
    return np.zeros(data.shape, dtype=np.float32)


def auto_midtone_stretch(
    data: np.ndarray,
    target_background: float = 0.25,
    black_point_percentile: float = 0.1,
    white_point_percentile: float = 99.9,
) -> np.ndarray:
    """Statistical Midtone Transfer Function (MTF).

    Calculates the exact midtone value needed to map the actual background
    median of `data` directly to `target_background` (e.g. 0.25).
    """
    source = normalise_image(data)
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

    # Solve PixInsight / Seti Astro style MTF equation for target median
    # m = (bg * (target - 1)) / (target * (2*bg - 1) - bg)
    midtone = (bg_median * (target - 1.0)) / (target * (2.0 * bg_median - 1.0) - bg_median)
    midtone = float(np.clip(midtone, 0.0001, 0.9999))

    num = (midtone - 1.0) * img
    den = (2.0 * midtone - 1.0) * img - midtone
    den = np.where(den == 0.0, 1e-7, den)

    return np.clip(num / den, 0.0, 1.0).astype(np.float32)


def midtone_transfer_function(
    data: np.ndarray, 
    target_median: float = 0.25, 
    shadow_clip: float = 0.001
) -> np.ndarray:
    """Statistical MTF stretch taking a target median value directly (0.0 to 1.0)."""
    # Simply delegate to the robust auto_midtone_stretch solver
    bp_pct = shadow_clip * 100.0
    return auto_midtone_stretch(
        data, 
        target_background=target_median, 
        black_point_percentile=bp_pct, 
        white_point_percentile=99.95
    )


def ghs_stretch(
    data: np.ndarray, 
    stretch_factor: float = 0.5, 
    symmetry_point: float = 0.005,
    local_intensity: float = 0.0
) -> np.ndarray:
    """Generalised Hyperbolic Stretch (GHS).
    
    - stretch_factor: [0.0 .. 1.0] maps logarithmically to D [0 .. 250,000]
    - symmetry_point: [0.0 .. 1.0] inflection point x0 (center of peak)
    - local_intensity: [0.0 .. 15.0] parameter b for focused contrast
    """
    img = normalise_image(data)
    
    if stretch_factor <= 0.0:
        return img

    # Map UI slider (0..1) to a realistic D scale (up to ~250,000)
    x = float(np.clip(stretch_factor, 0.0, 1.0))
    if x <= 0.0:
        return 0.0
    warped_x = np.power(x, 0.4)

    # Calculate actual linear background median
    bg_med = float(np.median(img))

    # If no symmetry point provided, lock x0 directly to the sky background level!
    if symmetry_point is None or symmetry_point <= 0.0:
        x0 = bg_med
    else:
        x0 = float(symmetry_point)

    # Clamp x0 safely so it never hits exact boundary limits
    x0 = np.clip(x0, 0.00001, 0.9999)
    
    D = np.power(10.0, warped_x * 6) - 1.0
    print(f"Stretch factor: {D:.2f}")
    #x0 = np.clip(symmetry_point, 0.00001, 0.9999)
    b = float(np.clip(local_intensity, 0.0, 15.0))

    # Generalized Hyperbolic Stretch core equation
    if b > 0.0:
        # Generalized form with local intensity parameter b
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


def arcsinh_stretch(data: np.ndarray, factor: float = 0.25, black_point: float = 0.0) -> np.ndarray:
    """Arcsinh stretch returning normalized float32."""
    img = normalise_image(data)
    s_factor = max(0.1, factor * 1000.0)
    bp = black_point * 0.1

    img_clipped = np.clip(img - bp, 0.0, 1.0)
    stretched = np.arcsinh(img_clipped * s_factor) / np.arcsinh(s_factor)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def log_stretch(data: np.ndarray, scaling_factor: float = 0.20) -> np.ndarray:
    """Logarithmic intensity stretch returning normalized float32."""
    img = normalise_image(data)
    a = 1.0 + (np.power(10.0, scaling_factor * 4.0) - 1.0)

    stretched = np.log1p(a * img) / np.log1p(a)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def root_stretch(data: np.ndarray, root_power: float = 0.20) -> np.ndarray:
    """Power-Law/Root Stretch returning normalized float32."""
    img = normalise_image(data)
    p = 1.0 + root_power * 19.0

    stretched = np.power(img, 1.0 / p)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def exp_stretch(data: np.ndarray, exponent_slope: float = 0.15) -> np.ndarray:
    """Exponential Stretch returning normalized float32."""
    img = normalise_image(data)
    b = 0.1 + exponent_slope * 29.9

    stretched = (1.0 - np.exp(-b * img)) / (1.0 - np.exp(-b))
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)