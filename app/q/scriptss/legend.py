import rasterio
import numpy as np

def calculate_quantile_breaks_skip_zeros(raster_path, num_quantiles):
    """
    Calculate quantile breaks for a raster file, excluding zero pixel values.

    Args:
        raster_path (str): Path to the raster file.
        num_quantiles (int): Number of quantiles to divide the data into.

    Returns:
        list: Quantile breakpoints.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the first band of the raster
        data = src.read(1)
        
        # Mask NoData values and filter out zeros
        data = data[(data != src.nodata) & (data != 0)]
        
        # Flatten the data array
        data = data.flatten()

    # Calculate quantile breakpoints
    quantile_breaks = np.percentile(data, np.linspace(0, 100, num_quantiles + 1))
    return quantile_breaks

# Path to the raster file
raster_path = r"C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\q\tiffs\32_bit.tiff"

# Number of quantiles
num_quantiles = 4  # Change this to the desired number of quantiles (e.g., 4, 5, etc.)

# Calculate quantile breaks
breaks = calculate_quantile_breaks_skip_zeros(raster_path, num_quantiles)

# Print the results
print("Quantile Breaks (excluding zero pixel values):")
for i, b in enumerate(breaks):
    print(f"Class {i + 1}: {b:.2f}")
