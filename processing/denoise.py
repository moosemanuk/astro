import numpy as np

def _b3_spline_kernel():
    """Returns a 1D 5-tap B3-spline filter kernel for a trous wavelet decomposition."""
    return np.array([1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0], dtype=np.float32)


def _convolve_2d_separable(image, kernel_1d, step=1):
    """
    Separable 2D convolution with step expansion (à trous) for 2D arrays.
    Uses mirror boundary padding to prevent edge artifacts.
    """
    # Expanded kernel with zeroes inserted (step - 1)
    k_len = len(kernel_1d)
    radius = (k_len // 2) * step

    # Pad image symmetrically
    padded = np.pad(image, radius, mode='reflect')

    # Row convolution
    row_conv = np.zeros_like(image, dtype=np.float32)
    for idx, w in enumerate(kernel_1d):
        offset = idx * step
        row_conv += w * padded[radius : radius + image.shape[0], offset : offset + image.shape[1]]

    # Column convolution
    padded_row = np.pad(row_conv, radius, mode='reflect')
    out = np.zeros_like(image, dtype=np.float32)
    for idx, w in enumerate(kernel_1d):
        offset = idx * step
        out += w * padded_row[offset : offset + image.shape[0], radius : radius + image.shape[1]]

    return out


def starlet_wavelet_denoise_channel(channel, num_scales=4, threshold_sigma=3.0, noise_floor=None):
    """
    Applies Starlet (À Trous) Wavelet Denoising to a single 2D float image array.
    
    Parameters:
    -----------
    channel : np.ndarray (2D float32)
        Single-channel 2D image data.
    num_scales : int
        Number of wavelet scales to decompose (3 to 5 is typical).
    threshold_sigma : float
        Multiplicative threshold factor (e.g., 3.0 means noise above 3*sigma is kept).
    noise_floor : float or None
        Estimated noise standard deviation. If None, estimated via Median Absolute
        Deviation (MAD) on the first high-frequency wavelet scale.
    """
    kernel = _b3_spline_kernel()
    current_approx = channel.copy().astype(np.float32)
    wavelet_details = []

    # 1. Forward Starlet Transform (À Trous Decomposition)
    for j in range(num_scales):
        step = 2**j
        next_approx = _convolve_2d_separable(current_approx, kernel, step=step)
        detail = current_approx - next_approx
        wavelet_details.append(detail)
        current_approx = next_approx

    # 2. Noise estimation if not provided (using MAD on 1st detail scale)
    if noise_floor is None:
        # MAD = median(|w1 - median(w1)|) / 0.6745
        mad = np.median(np.abs(wavelet_details[0] - np.median(wavelet_details[0])))
        noise_floor = mad / 0.6745

    # 3. Scale-dependent Soft Thresholding
    # Noise standard deviation scales down exponentially at coarser scales
    scale_factors = [1.0, 0.5, 0.25, 0.125, 0.0625]

    reconstructed = current_approx.copy()  # Start with the low-pass residual
    for j, detail in enumerate(wavelet_details):
        scale_sigma = scale_factors[min(j, len(scale_factors) - 1)]
        thresh = threshold_sigma * noise_floor * scale_sigma

        # Soft Thresholding: sign(x) * max(0, |x| - T)
        abs_detail = np.abs(detail)
        thresholded_detail = np.sign(detail) * np.maximum(0.0, abs_detail - thresh)
        reconstructed += thresholded_detail

    return reconstructed


def denoise_image(image_data, num_scales=4, threshold=3.0):
    """
    Main entry point compatible with main.py processing pipeline.
    Handles both 2D monochrome and 3D multi-channel (RGB) astronomical images.

    Parameters:
    -----------
    image_data : np.ndarray
        Astronomical image array matching your pyqtgraph/PyQt6 display shape.
    num_scales : int
        Wavelet scales (3 to 5).
    threshold : float
        Sigma cutoff for detail thresholding.

    Returns:
    --------
    np.ndarray
        Denoised image matching input shape and float32 dtype.
    """
    if image_data is None:
        return None

    data_32 = image_data.astype(np.float32)

    # Handle 2D Mono vs 3D RGB Arrays
    if data_32.ndim == 2:
        return starlet_wavelet_denoise_channel(
            data_32, num_scales=num_scales, threshold_sigma=threshold
        )
    elif data_32.ndim == 3:
        denoised = np.zeros_like(data_32)
        # Process each color channel independently
        for c in range(data_32.shape[2]):
            denoised[:, :, c] = starlet_wavelet_denoise_channel(
                data_32[:, :, c], num_scales=num_scales, threshold_sigma=threshold
            )
        return denoised
    else:
        raise ValueError(f"Unsupported image dimensions: {data_32.ndim}")