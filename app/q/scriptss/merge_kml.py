import geopandas as gpd

def calculate_kml_area(kml_file):
    """
    Calculate the total area of polygons in a KML file.

    Args:
        kml_file (str): Path to the KML file.

    Returns:
        float: Total area in square meters.
    """
    try:
        # Load the KML file as a GeoDataFrame
        gdf = gpd.read_file(kml_file)

        # Ensure the geometry is in a projected CRS (meters) for accurate area calculation
        gdf = gdf.to_crs(epsg=3857)  # EPSG:3857 is Web Mercator with units in meters

        # Calculate the total area of all geometries
        total_area = gdf['geometry'].area.sum()

        return total_area
    except Exception as e:
        print(f"Error processing the KML file: {e}")
        return None

# Path to the KML file
kml_file_path = r"C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\q\scriptss\merged.kml"

# Calculate the area
area = calculate_kml_area(kml_file_path)

if area is not None:
    print(f"Total area of the KML file: {area:.2f} square meters")
