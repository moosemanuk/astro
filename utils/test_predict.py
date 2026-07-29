import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
if not hasattr(np, "product"):
    np.product = np.prod

from astropy.io import fits
from n2v.models import N2V

# 1. Load FITS
hdul = fits.open("butterfly.fit")
image_data = hdul[0].data.astype(np.float32)
hdul.close()

if image_data.ndim == 3:
    image_data = image_data[0]

# 2. Normalize
img_min, img_max = np.min(image_data), np.max(image_data)
normalized_img = (image_data - img_min) / (img_max - img_min)

# 3. Load TRAINED model weights from disk
model = N2V(None, "my_fits_denoiser", basedir="saved_models")

# 4. Predict using correct axes ('YX')
denoised_norm = model.predict(normalized_img, axes='YX')
denoised_norm = np.squeeze(denoised_norm)

# 5. Restore original ADU range and save
denoised_fits_data = (denoised_norm * (img_max - img_min)) + img_min
fits.writeto("denoised_result.fit", denoised_fits_data, overwrite=True)

print("Finished! Saved denoised_result.fit")