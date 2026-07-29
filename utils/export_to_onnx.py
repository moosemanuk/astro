import os
# Force legacy Keras so TensorFlow loads the saved n2v weights without complaining
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
# Handle NumPy 2.0+ attribute mapping if needed
if not hasattr(np, "product"):
    np.product = np.prod

import tensorflow as tf
import tf2onnx
from n2v.models import N2V

print("1. Loading trained Noise2Void model weights from disk...")
# Load the model configuration and weights created during your training run
n2v_model = N2V(None, "my_fits_denoiser", basedir="saved_models")
keras_model = n2v_model.keras_model

print("2. Defining input tensor signature...")
# Input shape: [Batch_Size, Height, Width, Channels]
# Using 'None' for Height and Width allows dynamic image dimensions (e.g. 1000x1000 or 4000x3000)
input_signature = [
    tf.TensorSpec([None, None, None, 1], tf.float32, name="input")
]

print("3. Converting Keras model to ONNX graph...")
onnx_model, _ = tf2onnx.convert.from_keras(
    keras_model, 
    input_signature=input_signature, 
    opset=13
)

output_filename = "astro_denoiser.onnx"
print(f"4. Writing {output_filename} to disk...")
with open(output_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\nSUCCESS! Created '{output_filename}'.")
print("You can now distribute this ONNX file directly with your application.")