import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Load the FIT/S file

filename = "test.fit"

print(f"Loading FITS file: {filename}")

with fits.open(filename) as hdul:
    # Print the header information
    print("Header Information:")
    print(hdul.info())
    
    header = hdul[0].header
    data = hdul[0].data

# Print the header details
print("\n--- Data Inspection ---")
if data is None:
    print("No data found in the primary HDU.")
else:
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Mean: {np.mean(data)}")
    print(f"Standard Deviation: {np.std(data)}")
    print(f"Min pixel value: {np.min(data)}")
    print(f"Max pixel value: {np.max(data)}")

    print("\n--- Header Details ---")
    print(f"Object: {header.get('OBJECT', 'N/A')}")
    print(f"Date of Observation: {header.get('DATE-OBS', 'N/A')}")
    print(f"Exposure Time: {header.get('EXPTIME', 'N/A')} seconds")
    print(f"Filter: {header.get('FILTER', 'N/A')}")

    print("\nPlotting the data...")

    rgb_data = np.transpose(data, (1, 2, 0)) 
    rgb_data = rgb_data.astype(np.float32)
    rgb_data /= np.max(rgb_data)

    plt.imshow(rgb_data, origin='lower')
    plt.colorbar()
    plt.title("FITS Image")
    plt.show()
