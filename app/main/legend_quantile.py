import rasterio
import numpy as np

def calculate_quantile_breaks_skip_zeros(raster_path):
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
    quantile_breaks = np.percentile(data, np.linspace(0, 100, 6))  ## 5 is the number of breaks, so 5+1 = 6
    print(f"Quantile Breaks (NumPy): {quantile_breaks}")  # Before conversion
    print(f"Quantile Breaks (Python Floats): {[round(float(value), 2) for value in quantile_breaks]}")
    print([type(value) for value in quantile_breaks])
    return quantile_breaks

# # Path to the raster file
# raster_path = r"C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\q\tiffs\32_bit.tiff"


# # Calculate quantile breaks
# breaks = calculate_quantile_breaks_skip_zeros(raster_path)

# # Print the results
# print("Quantile Breaks (excluding zero pixel values):")
# for i, b in enumerate(breaks):
#     print(b)
