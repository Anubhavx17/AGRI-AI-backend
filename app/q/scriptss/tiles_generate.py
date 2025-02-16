


# import subprocess
# import os
# import sys
# from minio import Minio
# import tempfile
# import shutil

# # Step 1: MinIO configuration
# eos_client = Minio(
#     'objectstore.e2enetworks.net',
#     access_key='Q3VYHB4PV6GUW7CAGELM',  # Your access key
#     secret_key='2G52C4LTQ5CYKCJXLMJALCLGI1CUNNLDLRAHPENC',  # Your secret key
#     secure=True  # Use HTTPS
# )

# BUCKET_NAME = "dcmbucket"  # Replace with your bucket name
# MINIO_BASE_FOLDER = "tiles"  # The folder where all tiles will be uploaded

# # Step 2: Upload individual files to the bucket
# def upload_file_to_bucket(file_path, object_name):
#     try:
#         with open(file_path, 'rb') as file_obj:
#             file_size = os.fstat(file_obj.fileno()).st_size
#             eos_client.put_object(
#                 bucket_name=BUCKET_NAME,
#                 object_name=object_name,
#                 data=file_obj,
#                 length=file_size,
#                 content_type='application/octet-stream'
#             )
#         print(f"Uploaded: {object_name}")
#     except Exception as e:
#         print(f"Error uploading {object_name}: {e}")

# # Step 3: Recursively upload the tiles in the temp directory to the bucket under the `tiles` folder
# def upload_directory_to_bucket(directory, bucket_name, base_folder):
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             # Create the object name prefixed by the 'tiles' folder
#             object_name = os.path.join(base_folder, os.path.relpath(file_path, directory)).replace("\\", "/")
#             upload_file_to_bucket(file_path, object_name)

# # Step 4: Function to generate tiles in the temp directory using gdal2tiles
# def convert_geotiff_to_tiles(input_file, zoom_levels='0-13'):
#     # Create a temporary directory
#     temp_dir = tempfile.mkdtemp()

#     try:
#         # Use sys.executable to call the Python interpreter
#         python_executable = sys.executable
#         gdal2tiles_script = r"C:\Users\ANUBHAV\anaconda3\envs\agri_venv\Scripts\gdal2tiles.py"  # Update with the correct path

#         # Generate the tiles using gdal2tiles and store them in the temporary directory
#         subprocess.run([
#             python_executable,
#             gdal2tiles_script,
#             '-z', zoom_levels,
#             input_file,
#             temp_dir
#         ], check=True)

#         # After generating tiles, upload the entire temp directory to the bucket under the 'tiles' folder
#         upload_directory_to_bucket(temp_dir, BUCKET_NAME, MINIO_BASE_FOLDER)

#     finally:
#         # Clean up the temporary directory after uploading the tiles
#         shutil.rmtree(temp_dir)
#         print(f"Cleaned up temporary directory: {temp_dir}")

# # Example Usage:
# input_file = '8_bit.tiff'

# # Step 5: Generate the tiles and upload them directly to the 'tiles' folder in the MinIO bucket
# convert_geotiff_to_tiles(input_file)


import subprocess
import os
import sys


input_file, output_dir = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\output_data\REDSI_8bit.tiff', 'tiles_folder' ## this one is for making tiles uing vrt file
def convert_geotiff_to_tiles(input_file, output_dir, zoom_levels='0-13'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Use sys.executable to call the Python interpreter
    python_executable = sys.executable
    gdal2tiles_script = r"C:\Users\ANUBHAV\anaconda3\envs\agri_venv\Scripts\gdal2tiles.py"  # Update with the correct path

    subprocess.run([
        python_executable,
        gdal2tiles_script,
        '-z', zoom_levels,
        input_file,
        output_dir
    ])

# Example Usage:

convert_geotiff_to_tiles(input_file, output_dir)