import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from astropy.io import fits
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from n2v.models import N2VConfig, N2V
import numpy as np

# Monkey-patch NumPy 2.0+ for legacy n2v compatibility
if not hasattr(np, "product"):
    np.product = np.prod

# Load FITS data
hdul = fits.open("butterfly.fit")
image_data = hdul[0].data.astype(np.float32)
hdul.close()

if image_data.data.ndim == 3:
    image_data = image_data[0]

# Normalise the image
img_min = np.min(image_data)
img_max = np.max(image_data)
normalised_img = (image_data - img_min) / (img_max - img_min)

datagen = N2V_DataGenerator()

# Reshape array for N2V format: (1, Height, Width, 1)
reshaped_input = normalised_img[np.newaxis, ..., np.newaxis]

# Extract 64x64 sub-patches
patches = datagen.generate_patches_from_list([reshaped_input], shape=(64, 64))

# Split patches into Training (80%) and Validation (20%)
split = int(len(patches) * 0.8)
x_train = patches[:split]
x_val = patches[split:]

config = N2VConfig(
    x_train,
    unet_kern_size=3,
    train_steps_per_epoch=50,
    train_epochs=15,
    train_loss='mse',
    batch_norm=True,
    train_batch_size=64
)

# Initialise model
model_name = "my_fits_denoiser"
model = N2V(config, model_name, basedir="saved_models")

# START TRAINING! (1-3 mins depending on hardware)
model.train(x_train, x_val)

# Predict on full image array
denoised_normalised = model.predict(normalised_img, axes='YX')

# Remove extra dimensions to get back to 2D
denoised_normalised = np.squeeze(denoised_normalised)

# Scale back to your original FITS ADU range!
denoised_fits_data = (denoised_normalised * (img_max - img_min)) + img_min

