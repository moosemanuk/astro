# processing/pedestal.py
import numpy as np

def remove_pedestal(data: np.ndarray, percentile: float = 0.1, mode: str = "uniform") -> tuple[np.ndarray, list[float]]:
    """
    Calculates and subtracts baseline pedestal/offset.
    
    Parameters:
    - data: 2D or 3D numpy array.
    - percentile: Low percentile floor (default 0.1%).
    - mode: 'uniform' subtracts the same global pedestal across all channels (preserves color balance).
            'per_channel' subtracts each channel's offset individually.
            
    Returns:
    - corrected_data: Data with pedestal subtracted.
    - pedestal_values: List of subtracted pedestal values.
    """
    data_out = data.astype(np.float32, copy=True)
    pedestal_values = []

    if data_out.ndim == 3 and data_out.shape[2] in (3, 4):
        if mode == "uniform":
            # Find the lowest channel pedestal floor across the whole image
            channel_peds = [float(np.percentile(data_out[:, :, c], percentile)) for c in range(data_out.shape[2])]
            global_ped = min(channel_peds)
            
            # Subtract the same global scalar from all channels
            data_out = np.clip(data_out - global_ped, 0.0, None)
            pedestal_values = [global_ped] * data_out.shape[2]
        else:
            # Per-channel subtraction
            for c in range(data_out.shape[2]):
                ped_val = float(np.percentile(data_out[:, :, c], percentile))
                data_out[:, :, c] = np.clip(data_out[:, :, c] - ped_val, 0.0, None)
                pedestal_values.append(ped_val)
    else:
        # Mono Image
        ped_val = float(np.percentile(data_out, percentile))
        data_out = np.clip(data_out - ped_val, 0.0, None)
        pedestal_values.append(ped_val)

    return data_out, pedestal_values