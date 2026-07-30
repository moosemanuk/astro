import numpy as np
from scipy.ndimage import median_filter, uniform_filter, binary_dilation


def remove_hot_pixels(
    image: np.ndarray,
    threshold: float = 2.5,
    detect_chroma: bool = True,
    cluster_size: int = 2,
    progress_callback=None
) -> np.ndarray:
    """Multi-scale hot pixel, multi-pixel cluster, and walking noise suppressor.
    
    - threshold: Sigma sensitivity above local median background (2.0 to 3.5).
    - detect_chroma: Flag single-channel chromatic spikes (e.g. red/green/blue trails).
    - cluster_size: Dilation iterations (1 or 2) to catch contiguous warm pixels/trails.
    """
    cleaned = np.array(image, copy=True, order='C')
    is_rgb = (cleaned.ndim == 3 and cleaned.shape[2] == 3)

    if progress_callback:
        progress_callback(10)

    # --- PASS 1: Multi-scale Median Difference (Catches 1-4 px clusters) ---
    # Using a 5x5 median kernel prevents multi-pixel hot clusters from biasing the background estimate
    med5 = median_filter(cleaned, size=(5, 5, 1) if is_rgb else 5)
    
    # Estimate local deviation / MAD using uniform filters
    diff = np.abs(cleaned - med5)
    local_mad = uniform_filter(diff, size=(5, 5, 1) if is_rgb else 5) + 1e-6
    
    # Flag primary high-sigma anomalies
    hot_mask = diff > (threshold * local_mad)

    if progress_callback:
        progress_callback(40)

    # --- PASS 2: Chromatic Trail & Walking Noise Detection (RGB Only) ---
    if is_rgb and detect_chroma:
        local_mean_r = uniform_filter(cleaned[:, :, 0], size=5)
        local_mean_g = uniform_filter(cleaned[:, :, 1], size=5)
        local_mean_b = uniform_filter(cleaned[:, :, 2], size=5)

        std_r = np.sqrt(uniform_filter((cleaned[:, :, 0] - local_mean_r) ** 2, size=5)) + 1e-6
        std_g = np.sqrt(uniform_filter((cleaned[:, :, 1] - local_mean_g) ** 2, size=5)) + 1e-6
        std_b = np.sqrt(uniform_filter((cleaned[:, :, 2] - local_mean_b) ** 2, size=5)) + 1e-6

        # Spikes in one channel while other channels stay low
        spike_r = (cleaned[:, :, 0] - local_mean_r) > (threshold * std_r)
        spike_g = (cleaned[:, :, 1] - local_mean_g) > (threshold * std_g)
        spike_b = (cleaned[:, :, 2] - local_mean_b) > (threshold * std_b)

        hot_r = spike_r & (cleaned[:, :, 0] > (local_mean_g + 1.8 * std_g))
        hot_g = spike_g & (cleaned[:, :, 1] > (local_mean_r + 1.8 * std_r))
        hot_b = spike_b & (cleaned[:, :, 2] > (local_mean_r + 1.8 * std_r))

        chroma_mask = hot_r | hot_g | hot_b
        
        # Combine chromatic anomalies with core hot mask
        if hot_mask.ndim == 3:
            hot_mask = np.any(hot_mask, axis=2) | chroma_mask
        else:
            hot_mask = hot_mask | chroma_mask

    if progress_callback:
        progress_callback(70)

    # --- PASS 3: Morphological Dilation & Targeted Median Repair ---
    # Expand mask by cluster_size to capture blurred edges of trails/clusters
    if hot_mask.ndim == 3:
        hot_mask_2d = np.any(hot_mask, axis=2)
    else:
        hot_mask_2d = hot_mask

    dilated_mask = binary_dilation(hot_mask_2d, iterations=max(1, cluster_size))

    # Apply 5x5 median replacement only onto flagged pixel masks
    if is_rgb:
        for c in range(3):
            med_rep = median_filter(cleaned[:, :, c], size=5)
            cleaned[:, :, c][dilated_mask] = med_rep[dilated_mask]
    else:
        med_rep = median_filter(cleaned, size=5)
        cleaned[dilated_mask] = med_rep[dilated_mask]

    if progress_callback:
        progress_callback(100)

    return cleaned