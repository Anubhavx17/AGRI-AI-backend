from rio_tiler.colormap import cmap  # Import colormap definitions
import numpy as np
import rasterio
from PIL import Image

def apply_manual_colormap(input_8bit_tiff, output_colored_tiff, colormap_name="viridis"):
    """
    Manually apply a colormap to an 8-bit TIFF using NumPy.
    """
    # Step 1: Choose a colormap from rio_tiler.colormap.cmap
    colormap = cmap.get(colormap_name)

    if not colormap:
        raise ValueError(f"Colormap {colormap_name} not found in Rio Tiler")

    # Convert colormap to a simple dictionary of {int_value: (R, G, B, A)}
    simple_colormap = {key: tuple(map(int, value)) for key, value in colormap.items()}

    # Print the simple colormap for verification
    print(f"Formatted colormap '{colormap_name}': {list(simple_colormap.items())[:5]}...")

    # Step 2: Read the 8-bit TIFF using Rasterio and load it as a single-band array
    with rasterio.open(input_8bit_tiff) as src:
        if src.count != 1:
            raise ValueError(f"Expected 1 band, but got {src.count} bands in the input TIFF")

        # Read the first band
        data = src.read(1)

        # Debugging checks
        print(f"Data shape: {data.shape}")
        print(f"Data dimensions: {data.ndim}")
        print(f"Data type: {data.dtype}")

        # Prepare an empty RGBA image array with the same height and width as the data
        rgba_image = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)

        # Iterate through the pixel values and apply the corresponding color
        for pixel_value, color in simple_colormap.items():
            rgba_image[data == pixel_value] = color

        # Convert the NumPy array to an RGBA image using PIL
        final_image = Image.fromarray(rgba_image, 'RGBA')
        final_image.save(output_colored_tiff)
        print(f"Successfully applied {colormap_name} colormap to {input_8bit_tiff}, output saved at: {output_colored_tiff}")

# Example Usage:
if __name__ == "__main__":
    input_8bit_tiff = "8_bit.tiff"
    output_colored_tiff = "8_bit_colored_manual.tiff"
    colormap_name = "viridis"

    apply_manual_colormap(input_8bit_tiff, output_colored_tiff, colormap_name)

