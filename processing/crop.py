# processing/crop.py
import numpy as np


def crop_image(data, x, y, width, height):
  """Crops a 2D or 3D numpy array to the specified pixel coordinates.

  Parameters:
      data (np.ndarray): Image array (H, W) or (H, W, C)
      x (int): Starting X pixel coordinate (column)
      y (int): Starting Y pixel coordinate (row)
      width (int): Crop width in pixels
      height (int): Crop height in pixels
  """
  if data is None:
    return None

  # Get image dimensions (H, W)
  img_h, img_w = data.shape[0], data.shape[1]

  # Clamp coordinates safely within bounds
  x1 = max(0, int(x))
  y1 = max(0, int(y))
  x2 = min(img_w, x1 + max(1, int(width)))
  y2 = min(img_h, y1 + max(1, int(height)))

  # Slice numpy array
  if data.ndim == 3:
    return data[y1:y2, x1:x2, :].copy()
  else:
    return data[y1:y2, x1:x2].copy()