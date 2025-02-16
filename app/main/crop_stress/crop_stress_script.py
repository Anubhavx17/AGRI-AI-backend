import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentinelhub import CRS as sentinelCRS, BBox, DataCollection, MimeType, WcsRequest, CustomUrlParam
from s2cloudless import S2PixelCloudDetector
import rasterio
from rasterio.crs import CRS as rasterioCRS
from rasterio.enums import Resampling
import requests
from threading import Thread
import json
from rasterstats import zonal_stats
from scipy.constants import Stefan_Boltzmann
import rasterio.mask
import warnings
from flask import current_app as app, redirect
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from shapely.geometry import shape, Polygon, MultiPolygon
from app import db
from app.data_models.models import User, CropStressGraphModel,ResultTable
from app.data_models.schemas import CropStressGraphModelSchema
from osgeo import gdal
import time
from rio_viz.app import Client
# from .cog import cog_main
import sys
import io
import os
from minio import Minio
from minio.error import S3Error
import pandas as pd
from io import BytesIO
from sqlalchemy import cast, String,text
warnings.filterwarnings("ignore")
from rasterio.io import MemoryFile
# Create a Blueprint for the date fetching routes
crop_stress_bp = Blueprint('stress_calculation_testing', __name__)
crop_stress_graph_schema = CropStressGraphModelSchema(many=True)

eos_client = Minio(
    'objectstore.e2enetworks.net',
    access_key='Q3VYHB4PV6GUW7CAGELM',  # Your access key
    secret_key='2G52C4LTQ5CYKCJXLMJALCLGI1CUNNLDLRAHPENC',  # Your secret key
    secure=True  # Use HTTPS
)

BUCKET_NAME = "dcmbucket"  # Replace with your bucket name

def upload_file_to_bucket(file_obj, file_name):
    try:
        # Determine the file size based on the type of file_obj
        if isinstance(file_obj, io.BytesIO):
            file_size = file_obj.getbuffer().nbytes
        else:
            file_size = os.fstat(file_obj.fileno()).st_size
        
        # Upload the file to the bucket
        eos_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=file_name,
            data=file_obj,
            length=file_size,
            content_type='application/octet-stream'
        )
        
        # Construct the URL for the uploaded file
        file_url = f"https://objectstore.e2enetworks.net/{BUCKET_NAME}/{file_name}"
        print("files uploaded")
        return file_url
    except S3Error as e:
        print(f"Error occurred while uploading: {e}")
        return None

def update_or_upload_files(result_id, tiff_local_path, excel_local_path):
    """
    Uploads TIFF and Excel files to the storage and updates the ResultTable entry.

    Args:
    - result_id: The ID of the result entry in the database.
    - tiff_local_path: The local path to the TIFF file.
    - excel_local_path: The local path to the Excel file.

    Returns:
    - A tuple containing the URLs for the TIFF and Excel files.
    """
    # Construct filenames based on the result_id
    tiff_filename = f"{result_id}.tiff"
    excel_filename = f"{result_id}.xlsx"

    # Upload TIFF file
    with open(tiff_local_path, "rb") as tiff_file:
        tiff_url = upload_file_to_bucket(tiff_file, tiff_filename)

    # Upload Excel file
    with open(excel_local_path, "rb") as excel_file:
        excel_url = upload_file_to_bucket(excel_file, excel_filename)

    return tiff_url, excel_url

def create_result_entry(user_id, tiff_local_path, excel_local_path,tiff_min_max,redsi_min_max_str, selected_date, selected_parameter, geojson,project_id):
    # Convert the numpy array to a string
    tiff_min_max_str = np.array2string(tiff_min_max, separator=',')

    # geojson_str = json.dumps(geojson) if isinstance(geojson, dict) else geojson

    # Check if an entry with the same selected_date, selected_parameter, and geojson already exists
    existing_result = ResultTable.query.filter(
        ResultTable.selected_date == selected_date,
        ResultTable.selected_parameter == selected_parameter,
        ResultTable.project_id == project_id,
        # cast(ResultTable.geojson, String) == geojson_str  # Compare with the JSON string representation
    ).first()
    
    if existing_result:
        # If an existing entry is found, update its min_max field
        # existing_result.tiff_min_max = tiff_min_max_str

        # # Update the TIFF and Excel URLs using the new function
        # tiff_url, excel_url = update_or_upload_files(existing_result.id, tiff_local_path, excel_local_path)
        # existing_result.tiff = tiff_url
        # existing_result.excel = excel_url
        # existing_result.redsi_min_max = redsi_min_max_str
        # # Commit the updates
        # db.session.commit()
        # print("Entry exists, updated tiff_min_max, TIFF, and Excel.")
        return existing_result.id
    
    # Step 1: Create an entry in the ResultTable
    new_result = ResultTable(user_id=user_id, tiff='', excel='', tiff_min_max=tiff_min_max_str,redsi_min_max=redsi_min_max_str, selected_date=selected_date, selected_parameter=selected_parameter,
                              geojson =geojson,project_id=project_id)  # Store as a string
    db.session.add(new_result)
    db.session.commit()  # Commit to get the auto-incremented ID

    # Step 2: Use the ID to construct filenames
    result_id = new_result.id
    tiff_filename = f"{result_id}.tiff"
    excel_filename = f"{result_id}.xlsx"

    # Step 3: Upload TIFF file
    with open(tiff_local_path, "rb") as tiff_file:
        tiff_url = upload_file_to_bucket(tiff_file, tiff_filename)
        
    # Step 4: Upload Excel file
    with open(excel_local_path, "rb") as excel_file:
        excel_url = upload_file_to_bucket(excel_file, excel_filename)

    # Step 5: Update the ResultTable entry with the URLs
    new_result.tiff = tiff_url
    new_result.excel = excel_url
    db.session.commit()
    print("result table entry created")
    return result_id

def get_presigned_url(bucket_name, object_name, expiration=timedelta(hours=1)):
    try:
        # Generate a presigned URL valid for 1 hour (default)
        presigned_url = eos_client.presigned_get_object(bucket_name, object_name, expires=expiration)
        return presigned_url
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None

# rioclient = None
# min_max_str = None

# def load_new_tiff(tiff_path, tiff_min_max_str):
#     global min_max_str, rioclient
#     min_max_str = tiff_min_max_str

#     # Restart the Rio-Viz client with the new TIFF
#     # if rioclient:
#     #     rioclient.shutdown()
    
#     rioclient = Client(tiff_path)
#     time.sleep(1)
#     print(" in load new tiff", rioclient)
#     return {"message": "TIFF loaded successfully"}

@crop_stress_bp.route("/tiles/<int:z>/<int:x>/<int:y>.png")
def serve_tiles(z, x, y):
    """
    Serve tiles using the dynamically selected rio-viz client.
    """
    selected_date = request.args.get("selected_date")
    if not selected_date:
        return jsonify({"error": "Missing selected_date query parameter"}), 400

    response = requests.post("http://127.0.0.1:5001/get_client_port", json={"selected_date": selected_date})
    if response.ok:
        client_info = response.json()
        port = client_info.get("port")
        tiff_min_max = client_info.get("tiff_min_max")
        colormap = "hsv"
        rio_viz_url = (f"http://127.0.0.1:{port}/tiles/{z}/{x}/{y}.png"
                       f"?nodata=0&colormap_name={colormap}&rescale={tiff_min_max}")

        return redirect(rio_viz_url)
    else:
        return jsonify({"error": "Failed to fetch client info"}), 500

@crop_stress_bp.route('/get_tiff_data', methods=['POST'])
@jwt_required()
def get_tiff_data():
    start_time = time.time()  # Start the timer
    data = request.get_json()
    user_id = get_jwt_identity()
    project_id = data.get('project_id')
    tiff_data = []
    results = ResultTable.query.filter_by(user_id=user_id, project_id=project_id).all()
    
    for result in results:
        presigned_tiff_url = get_presigned_url(BUCKET_NAME, f"{result.id}.tiff")
        tiff_data.append({
            'selected_date': result.selected_date,
            'tiff_url': presigned_tiff_url,
            'tiff_min_max': result.tiff_min_max,
            'legend_quantile': result.legend_quantile
        })

    # clients.append({result.selected_date: Client(presigned_tiff_url)})
        # legend_quantile = calculate_quantile_breaks_skip_zeros(presigned_tiff_url)
        # print(result.selected_date, " -> ",result.tiff_min_max, " -> ", str(legend_quantile))  # Debugging output
       
        # result.legend_quantile = str(legend_quantile)
        # Append the result to the response

    response = requests.post("http://127.0.0.1:5001/initialize", json=tiff_data)
    if response.ok:
        print("Clients initialized successfully.")
    else:
        print("Error initializing clients:", response.json())
    
    end_time = time.time()  # End the timer
    print(f"Time taken for /get_tiff_data endpoint: {end_time - start_time:.2f} seconds")
    
    return jsonify(tiff_data)

@crop_stress_bp.route("/api/load_tiff", methods=["POST"])
@jwt_required()
def get_tiff_path():
    data = request.get_json()
    selected_date = data.get("selected_date")
    response = requests.post("http://127.0.0.1:5001/get_client_port", json={"selected_date": selected_date})
    if response.ok:
        client_info = response.json()
        port = client_info.get("port")
        tiff_min_max = client_info.get("tiff_min_max")
        tile_url = (
            f"http://127.0.0.1:{port}/tiles/{{z}}/{{x}}/{{y}}.png"
            f"?nodata=0&colormap_name=hsv&rescale={tiff_min_max}"
        )
        print(tile_url, tiff_min_max)
        return jsonify({"tile_url": tile_url, "tiff_min_max": tiff_min_max})
    else:
        return jsonify({"error": "Failed to fetch client info"}), 500


### MAIN FUNCTION
@crop_stress_bp.route('/stress_calculation_testing', methods=['POST'])
@jwt_required()
def main():

    data = request.get_json()
    user_id = get_jwt_identity()
    geojson_data,input_date, crop = get_input_data(data) ## tweaking the input data
    project_id = data.get('project_id')
    sentinel_data, transform, width, height, dtype,bbox = prepare_sentinel_data(geojson_data, input_date)

    # Calculate CY (Crop Yield) and REDSI (RedEdge Stress Index) and Cloud Mask Calculation
    cy_mean_dict,cy = process_cy(geojson_data, input_date, crop, sentinel_data, transform, width, height, dtype,bbox)
    redsi_mean_dict,redsi,tiff_min_max = process_redsi(geojson_data, sentinel_data, transform, width, height, dtype)
    masks_mean_dict = process_cloud_mask(geojson_data, sentinel_data, transform, width, height, dtype)

   
    redsi_path = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\output_data\REDSI.tiff'
    excel_path = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\output_data\excel.xlsx'

    # load_new_tiff(redsi_path,tiff_min_max)

    # Perform inferencing and generate the final Excel file
 
    temp_df,redsi_min_max = generate_final_excel(geojson_data, input_date, crop, cy, redsi, redsi_mean_dict, masks_mean_dict)

    result_id = create_result_entry(user_id, redsi_path, excel_path, tiff_min_max, redsi_min_max,data.get('date'), data.get('selectedParameter'), data.get('GeojsonData'),project_id)

    # output_8bit_path = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\output_data\REDSI_8bit.tiff'
    # color_table = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\crop_stress\color.txt'
    
    #  # Step 1: Convert 32-bit TIFF to 8-bit TIFF
    # convert_32bit_to_8bit(redsi_path, output_8bit_path,color_table)
    # # Step 2: Generate the tiles and upload them directly to the 'tiles' folder in the MinIO bucket
    # convert_geotiff_to_tiles(output_8bit_path,result_id)
    # # run_cog_conversion()
  
    # Save the temp_df to db
    # save_temp_df_to_db(temp_df,result_id)

   # Fetch the newly created ResultTable entry to return the URLs
    result_entry = ResultTable.query.get(result_id)
    
    if not result_entry:
        return jsonify({ "dataSaved": False,"error": "Could not find the result entry, pls try again"}), 200
    
    # convert_tiff_to_signed_8bit(redsi_path, output_8bit_path, tiff_min_max)

    tiff_presigned_url = get_presigned_url(BUCKET_NAME, f"{result_entry.id}.tiff")
    excel_presigned_url = get_presigned_url(BUCKET_NAME, f"{result_entry.id}.xlsx")

    # Prepare the response data
    response_data = {
        "dataSaved": True,
        "id": result_entry.id,
        "user_id": result_entry.user_id,
        "tiff_url": tiff_presigned_url,
        "excel_url": excel_presigned_url,
        "tiff_min_max" : result_entry.tiff_min_max,
        "redsi_min_max" :result_entry.redsi_min_max,
        "geojson" :result_entry.geojson,
         "legend_quantile":result_entry.legend_quantile
    }
    print("response data sent")
    return jsonify(response_data), 200

def clean_json_data(data):
    """
    Recursively clean the JSON data by replacing inf, -inf, and NaN with None.
    """
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, float) and (np.isinf(data) or np.isnan(data)):
        return None
    return data

@jwt_required()
def save_results(data,result_id):
    """
    Saves the result data into the CropStressGraphModel.

    Args:
    - data: A dictionary containing the fields required to save to the CropStressGraphModel.

    Returns:
    - A success message.
    """
    existing_entry = CropStressGraphModel.query.filter_by(
        unique_farm_id=data.get('unique_farm_id'),
        selected_date=data.get('selected_date'),
        selected_parameter=data.get('selected_parameter'),
        result_id = result_id
    ).first()

    if existing_entry:
        # If an entry exists, return a message indicating that it already exists
        print("Entry already exists, no new record created")
        return jsonify({"msg": "Entry already exists, no new record created", "projectSaved": False}), 201


    current_user_id = get_jwt_identity()
    cleaned_result_details = clean_json_data(data.get('result_details'))
    cleaned_geojson = clean_json_data(data.get('geojson'))
    geojson_str = json.dumps(cleaned_geojson)

    # Convert cleaned_result_details to a JSON string
    result_details_str = json.dumps(cleaned_result_details)

    
    stress_result = CropStressGraphModel(
        unique_farm_id=data.get('unique_farm_id'),
        user_id=current_user_id,
        # geojson=data.get('geojson'),
        geojson=geojson_str,
        selected_date=data.get('selected_date'),
        selected_parameter=data.get('selected_parameter'),
        # result_details=data.get('result_details'),
        result_details=result_details_str,  
        result_id=result_id
    )

    db.session.add(stress_result)
    db.session.commit()
    return jsonify({"msg": "Project saved successfully!", "projectSaved": True}), 201

def centroidForZoom(geojson_data):
    """
    Calculate the centroid of a given GeoJSON feature.
    
    Args:
    - geojson_data: A dictionary containing the GeoJSON feature data.

    Returns:
    - A dictionary containing the latitude and longitude of the centroid, or None if not applicable.
    """
    try:
        # Convert the GeoJSON geometry to a shapely geometry object
        geometry = shape(geojson_data['geometry'])

        # Check if the geometry is valid and not empty
        if geometry.is_valid and not geometry.is_empty:
            # Calculate the centroid
            centroid = geometry.centroid
            return {'latitude': centroid.y, 'longitude': centroid.x}
        else:
            print("Warning: Invalid or empty geometry found.")
            return None

    except Exception as e:
        print(f"Error processing geometry. Error: {e}")
        return None

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
    quantile_breaks = [round(float(value), 2) for value in quantile_breaks]
    return quantile_breaks

@crop_stress_bp.route('/get_inference_data', methods=['POST'])
def fetch_inference_data():
    data = request.get_json()
    result_id = data.get('result_id')

    if result_id is None:
        return jsonify({"error": "Missing required parameter 'result_id'"}), 400

    try:
        query = """
            SELECT 
                CASE
                    WHEN result_details LIKE '%INFERENCE: Presence of Cloud%' THEN 'Presence of Cloud'
                    WHEN result_details LIKE '%INFERENCE: Severe Crop Stress%' THEN 'Severe Crop Stress'
                    WHEN result_details LIKE '%INFERENCE: No Crop Stress%' THEN 'No Crop Stress'
                END AS inference,
                COUNT(*) AS count,
                ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM public.crop_stress_graph_model WHERE result_id = :result_id)), 2) AS percentage,
                ARRAY_AGG(unique_farm_id) AS unique_farm_id
            FROM 
                public.crop_stress_graph_model
            WHERE 
                result_id = :result_id
            GROUP BY 
                inference;
        """
        
        # Execute the query
        result = db.session.execute(text(query), {"result_id": result_id})
        
        # Extract column names and rows, convert to list of dictionaries
        keys = result.keys()
        data = [dict(zip(keys, row)) for row in result]

        return jsonify(data), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@crop_stress_bp.route('/fetch_graph_details', methods=['POST'])
@jwt_required()
def fetch_results():
    """
    Fetches the results based on the unique_farm_id.
    Returns:
    - A JSON containing the selected date, parameter, geojson, and result details.
    """
    # load_new_tiff(r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\output_data\REDSI.tiff')
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Get the unique farm ID from the request
    unique_farm_id = data.get('unique_farm_id')

    # Query the database for the requested farm and the current user
    results = CropStressGraphModel.query.filter_by(
        unique_farm_id=unique_farm_id,
        user_id=current_user_id
    ).all()

    geojson = results[0].geojson if results else None
    centroid = centroidForZoom(geojson)
    print(centroid)
    if not results:
        return jsonify({"msg": "No results found for this farm", "results": []}), 404

    # Initialize the schema
    crop_stress_schema = CropStressGraphModelSchema(many=True)

    # Serialize the query results
    results_serialized = crop_stress_schema.dump(results)
    # Return the serialized results
    return jsonify({"msg": "Results fetched successfully", "results": results_serialized, "centroid":centroid}), 200


def save_temp_df_to_db(temp_df,result_id):
    """
    Iterates through temp_df and saves each row to the database using save_results.
    
    Args:
    - temp_df: A DataFrame containing the results to be saved in the database.
    
    Returns:
    - Success message after all rows are saved.
    """
   
    for index, row in temp_df.iterrows():
        # Prepare data for saving
        data = {
            'unique_farm_id': row['Unique_ID'],
            'geojson': json.loads(row['geojson_data']),  # Convert geojson_data string to JSON
            'selected_date': row['Date'],
            'selected_parameter': row['Parameter'],
            'result_details': row['Result'],
        }

        # Save the row using the save_results function
        save_results(data, result_id)
    print('project saved')
    return "done"


def dict_to_gdf(geojson_data):
    """
    Converts a GeoJSON-like dictionary into a GeoDataFrame, keeping the same structure.
    Ensures properties come first, followed by the geometry.
    """
    # Create the GeoDataFrame, ensuring the properties are preserved as columns
    gdf = gpd.GeoDataFrame.from_features(geojson_data['features'])

    # Ensure that geometry is the last column and properties come first
    columns = [col for col in gdf.columns if col != 'geometry'] + ['geometry']
    gdf = gdf[columns]  # Reorder the columns
    
    return gdf


def get_config():
    ### LOADING SENTINEL HUB CONFIGURATION
    instance_id = 'f3847df2-896e-4c88-aa4e-f7177a08935b'
    client_id = app.config['SENTINEL_CLIENT_ID']
    client_secret = app.config['SENTINEL_CLIENT_SECRET']

    ### SENTINEL HUB INSTANCE INFORMATION
    from sentinelhub import SHConfig
    config = SHConfig()
    config.instance_id = instance_id
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.save()
    return config

### DATE INPUT 
def get_input_data(data):
    
    input_date =  datetime.strptime(data['date'], '%d/%m/%Y').strftime('%Y-%m-%d')
    crop = data.get('selectedCrop')[0]  
    # print(data.get('GeojsonData'))
    geojson_data = dict_to_gdf(data.get('GeojsonData')) ## make it a gdf
    # print('Input Date - ', input_date)
    # print('Crop - ', crop)
    # print("Input Data fetched")
    return geojson_data,input_date, crop


### MOSAIC FUNCTION
def mosaic_tiff(tiffs):
    tiffs = [np.nan_to_num(arr) for arr in tiffs]
    height, width = tiffs[0].shape
    mosaic = np.full(shape = (height, width), fill_value = float('-inf'), dtype = tiffs[0].dtype)
    for tiff in tiffs:
        mosaic = np.maximum(mosaic, tiff)
    print("Mosaic done")
    return mosaic


### DEFINING BOUNDING BOX
def dimensions(geojson_data):
    minx, miny, maxx, maxy = geojson_data.total_bounds
    extent = [minx, miny, maxx, maxy]
    bbox = BBox(bbox = extent, crs = sentinelCRS.WGS84)
    print("Bounding Box created")
    return bbox, extent


### SENTINELHUB API WCS REQUEST FOR BAND 1 REFLECTANCE LAYER
def band1_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B01',
        layer = 'B01',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band1_ref = mosaic_tiff(wcs_true_color_img)
    print("Band 1 Reflectance fetched")
    return band1_ref


### SENTINELHUB API WCS REQUEST FOR BAND 2 REFLECTANCE LAYER
def band2_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B02',
        layer = 'B02',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band2_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 2 Reflectance fetched")
    
    return band2_ref


### SENTINELHUB API WCS REQUEST FOR BAND 3 REFLECTANCE LAYER
def band3_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B03',
        layer = 'B03',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band3_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 3 Reflectance fetched")
    
    return band3_ref


### SENTINELHUB API WCS REQUEST FOR BAND 4 REFLECTANCE LAYER
def band4_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B04',
        layer = 'B04',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band4_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 4 Reflectance fetched")
    
    return band4_ref


### SENTINELHUB API WCS REQUEST FOR BAND 5 REFLECTANCE LAYER
def band5_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B05',
        layer = 'B05',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band5_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 5 Reflectance fetched")
    
    return band5_ref


### SENTINELHUB API WCS REQUEST FOR BAND 6 REFLECTANCE LAYER
def band6_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B06',
        layer = 'B06',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band6_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 6 Reflectance fetched")
    
    return band6_ref


### SENTINELHUB API WCS REQUEST FOR BAND 7 REFLECTANCE LAYER
def band7_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B07',
        layer = 'B07',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band7_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 7 Reflectance fetched")
    
    return band7_ref


### SENTINELHUB API WCS REQUEST FOR BAND 8 REFLECTANCE LAYER
def band8_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B08',
        layer = 'B08',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band8_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 8 Reflectance fetched")
    
    return band8_ref


### SENTINELHUB API WCS REQUEST FOR BAND 8A REFLECTANCE LAYER
def band8a_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B8A',
        layer = 'B8A',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band8a_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 8A Reflectance fetched")
    
    return band8a_ref


### SENTINELHUB API WCS REQUEST FOR BAND 9 REFLECTANCE LAYER
def band9_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B09',
        layer = 'B09',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band9_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 9 Reflectance fetched")
    
    return band9_ref


### SENTINELHUB API WCS REQUEST FOR FAPAR LAYER
### FAPAR --> FRACTION OF ABSORBED PHOTOSYNTHETICALLY ACTIVE RADIATION
def fapar_layer_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/FAPAR',
        layer = 'FAPAR',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    FAPAR = mosaic_tiff(wcs_true_color_img)

    print("FAPAR fetched")
    
    return FAPAR


### SENTINELHUB API WCS REQUEST FOR LSWI LAYER
### LSWI --> LAND SURFACE WATER INDEX
def lswi_layer_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/LSWI',
        layer = 'LSWI',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = get_config()
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    LSWI = mosaic_tiff(wcs_true_color_img)

    print("LSWI fetched")
    
    return LSWI


### SENTINEL DATA DICTINOARY
def sentinel_data_dict(bbox, input_date):
    band1_ref = band1_reflectance_call(bbox, input_date)
    band2_ref = band2_reflectance_call(bbox, input_date)
    band3_ref = band3_reflectance_call(bbox, input_date)
    band4_ref = band4_reflectance_call(bbox, input_date)
    band5_ref = band5_reflectance_call(bbox, input_date)
    band6_ref = band6_reflectance_call(bbox, input_date)
    band7_ref = band7_reflectance_call(bbox, input_date)
    band8_ref = band8_reflectance_call(bbox, input_date)
    band8a_ref = band8a_reflectance_call(bbox, input_date)
    band9_ref = band9_reflectance_call(bbox, input_date)
    FAPAR = fapar_layer_call(bbox, input_date)
    LSWI = lswi_layer_call(bbox, input_date)

    sentinel_data = {
        'BAND1_REF' : band1_ref,
        'BAND2_REF' : band2_ref,
        'BAND3_REF' : band3_ref,
        'BAND4_REF' : band4_ref,
        'BAND5_REF' : band5_ref,
        'BAND6_REF' : band6_ref,
        'BAND7_REF' : band7_ref,
        'BAND8_REF' : band8_ref,
        'BAND8A_REF' : band8a_ref,
        'BAND9_REF' : band9_ref,
        'FAPAR' : FAPAR,
        'LSWI' : LSWI
    }

    return sentinel_data


### CLOUD MASK CALCULATIONS
def cloud_detector(sentinel_data):
    band1_ref = sentinel_data['BAND1_REF']
    band2_ref = sentinel_data['BAND2_REF']
    band3_ref = sentinel_data['BAND3_REF']
    band4_ref = sentinel_data['BAND4_REF']
    band5_ref = sentinel_data['BAND5_REF']
    band6_ref = sentinel_data['BAND6_REF']
    band7_ref = sentinel_data['BAND7_REF']
    band8_ref = sentinel_data['BAND8_REF']
    band8a_ref = sentinel_data['BAND8A_REF']
    band9_ref = sentinel_data['BAND9_REF']

    bands = [band1_ref, band2_ref, band3_ref, band4_ref, band5_ref,
             band6_ref, band7_ref, band8_ref, band8a_ref, band9_ref]
    layer_stack = np.stack(bands, axis = -1)

    cloud_detector = S2PixelCloudDetector(threshold = 0.4, average_over = 4, dilation_size = 2, all_bands = False)
    cloud_prob = cloud_detector.get_cloud_probability_maps(layer_stack[np.newaxis, ...])
    cloud_mask = cloud_detector.get_cloud_masks(layer_stack[np.newaxis, ...])

    print("Cloud Probabilities and Mask generated")

    return cloud_prob, cloud_mask


### WATER STRESS SCALAR (W) CALCULATIONS
def w_calc(sentinel_data):
    LSWI = sentinel_data['LSWI']
    LSWI_max = np.amax(LSWI)
    w = (1 - LSWI) / (1 + LSWI_max)

    print("Water Stres Scalar calculated")
    
    return w


### LIGHT USE EFFICIENCY (LUE) CALCULATIONS
def lue_calc(sentinel_data):
    e0 = 3.22          # MAXIMUM VALUE OF LUE
    w = w_calc(sentinel_data)
    LUE = e0 * w

    print("LUE calculated")
    
    return LUE


### GLOBAL HORIZONTAL IRRADIANCE CALCULATIONS
def ghi_calc(input_date, bbox):
    lat = bbox.middle[1]
    lon = bbox.middle[0]
    API_KEY = app.config['OPENWEATHER_API_KEY']
    response = requests.get(f"https://api.openweathermap.org/energy/1.0/solar/data?lat={lat}&lon={lon}&date={input_date}&appid={API_KEY}")
    data = response.json()
    GHI = 0
    for i in data["irradiance"]["daily"]:
        GHI = i['clear_sky']['ghi']
    GHI = (GHI * 3.6) / 1000

    print("GHI calculated")
    
    return GHI


### NET PRIMARY PRODUCTIVITY (NPP) CALCULATIONS
def npp_calc(input_date, sentinel_data, bbox):
    FAPAR = sentinel_data['FAPAR']
    GHI = ghi_calc(input_date, bbox)
    LUE = lue_calc(sentinel_data)
    # print(FAPAR, " - Fapar")
    # print(GHI, " - GHI")
    # print(LUE, " - LUE")
    NPP = FAPAR * GHI * 0.5 * LUE
    nodata = 0
    NPP[NPP < 0] = nodata

    # print("NPP calculated")
    
    return NPP


### HARVEST INDEX DICTIONARY
def harvest_index_dict(crop):
    harvest_index = {
        "Sugarcane" : 0.9,    ### originally 0.8
        "Potato" : 0.2,
        "Cotton" : 0.008,
        "Wheat" : 0.01,
        "Corn" : 0.045,
        "Chilli" : 0.017
    }
    
    return harvest_index[crop]


### CROP YIELD CALCULATIONS
def cy_calc(input_date, crop, sentinel_data, bbox):
    NPP = npp_calc(input_date, sentinel_data, bbox)
    HI = harvest_index_dict(crop)      
    cy = NPP * HI * 10
   
    return cy


### REDSI CALCULATIONS
def redsi_calc(sentinel_data):
    band4_ref = sentinel_data['BAND4_REF']
    band5_ref = sentinel_data['BAND5_REF']
    band7_ref = sentinel_data['BAND7_REF']
    # 665nm --> wavelength of band4 reflectance
    # 705nm --> wavelength of band5 reflectance
    # 783nm --> wavelength of band7 reflectance
    redsi = (((705 - 665) * (band7_ref - band4_ref)) - ((783 - 665) * (band5_ref - band4_ref))) / (2 * band4_ref)
    
    print('REDSI Calculated')

    return redsi


### GROWTH PHASE CALCULATION
def growth_phase_calc(crop, input_date, geojson_data):
    def parse_date(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d')

    input_date = parse_date(input_date)  
    stage_dict = {}
    for i in range(len(geojson_data)):
        growth_phase_dict = {}
        planting_date = geojson_data['PLANT_DAY'][i]
        planting_date = datetime.strptime(planting_date, "%Y-%m-%d").date()
        if crop == 'Sugarcane':
            if '2023-09-01' <= planting_date.strftime("%Y-%m-%d") <= '2023-11-30':    ## AUTUMN PLANTS
                growth_phase_dict = {
                    (f'{planting_date}', f'{planting_date + timedelta(days = 50)}') : 'Germination',
                    (f'{planting_date + timedelta(days = 50)}', f'{planting_date + timedelta(days = 120)}') : 'Tillering',
                    (f'{planting_date + timedelta(days = 120)}', f'{planting_date + timedelta(days = 175)}') : 'Grand Growth',
                    (f'{planting_date + timedelta(days = 175)}', f'{planting_date + timedelta(days = 250)}') : 'Summer',
                    (f'{planting_date + timedelta(days = 250)}', f'{planting_date + timedelta(days = 345)}') : 'Maturity'
                }
            elif '2024-01-01' <= planting_date.strftime("%Y-%m-%d") <= '2024-03-31':    ## SPRING PLANTS
                growth_phase_dict = {
                    (f'{planting_date}', f'{planting_date + timedelta(days = 45)}') : 'Germination',
                    (f'{planting_date + timedelta(days = 45)}', f'{planting_date + timedelta(days = 115)}') : 'Tillering',
                    (f'{planting_date + timedelta(days = 115)}', f'{planting_date + timedelta(days = 185)}') : 'Monsoon',
                    (f'{planting_date + timedelta(days = 185)}', f'{planting_date + timedelta(days = 218)}') : 'Grand Growth',
                    (f'{planting_date + timedelta(days = 218)}', f'{planting_date + timedelta(days = 250)}') : 'Later Grand Growth',
                    (f'{planting_date + timedelta(days = 250)}', f'{planting_date + timedelta(days = 360)}') : 'Maturity'
                }
    
        for date_range, stage in growth_phase_dict.items():
            start_date, end_date = map(parse_date, date_range)
            if start_date <= input_date <= end_date:
                stage_dict[geojson_data['PLOT_NO'][i]] = stage

    print("Growth stage dictionary created")

    return stage_dict


### RATOON DICTIONARY CALCULATIONS
def ratoon_dict_calc(geojson_data):
    ratoon_dict = {}
    for i in range(len(geojson_data)):
        ratoon_dict[geojson_data['PLOT_NO'][i]] = geojson_data['TYPE'][i]

    return ratoon_dict


### FINAL INFERENCING CALCULATIONS
def inferencing(crop, input_date, geojson_data, redsi_mean_dict, masks_mean_dict):
    ratoon_dict = ratoon_dict_calc(geojson_data)
    stage_dict = growth_phase_calc(crop, input_date, geojson_data)
    inference = {}
    sub_inference = {}
    for key, value in redsi_mean_dict.items():
        if masks_mean_dict[key] == 0:
            if ratoon_dict[key] == 'Spring':
                if stage_dict[key] == 'Germination':
                    if value >= 0:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 0 and value >= -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -10 and value >= -35:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < -35:
                        inference[key] = 'Severe Stress'
                        sub_inference[key] = 'Red Rot Stress'

                elif stage_dict[key] == 'Tillering':
                    if value >= 0:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 0 and value >= -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < -20 and value >= -40:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -40:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

                elif stage_dict[key] == 'Monsoon':
                    if value >= 40:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 40 and value >= 30:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < 30 and value >= 20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < 20 and value >= 0:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

                elif stage_dict[key] == 'Grand Growth':
                    if value >= 20:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 20 and value >= 0:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < 0 and value >= -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

                elif stage_dict[key] == 'Later Grand Growth':
                    if value >= 10:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 10 and value >= -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < -10 and value >= -30:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -30:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

                elif stage_dict[key] == 'Maturity':
                    if value >= -10:
                        inference[key] = 'No Crop Stres'
                        sub_inference[key] = 'NA'
                    elif value < -10 and value >= -15:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < -15 and value >= -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

            
            elif ratoon_dict[key] == 'Spring_Ratoon':
                if stage_dict[key] == 'Germination':
                    if value >= 0:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 0:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'NA'

                elif stage_dict[key] == 'Tillering':
                    if value >= 0:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 0:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'NA'

                elif stage_dict[key] == 'Monsoon':
                    if value >= 10:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'NA'

                elif stage_dict[key] == 'Grand Growth':
                    if value >= 10:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'NA'

                elif stage_dict[key] == 'Maturity':
                    if value >= 0:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 0:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'NA'
                

            elif ratoon_dict[key] == 'Autumn' or ratoon_dict[key] == 'Autumn_Ratoon':
                if stage_dict[key] == 'Germination':
                    if value >= 0:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 0 and value >= -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -10 and value >= -35:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < -35:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

                elif stage_dict[key] == 'Tillering':
                    if value >= 0:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 0 and value >= -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -10 and value >= -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

                elif  stage_dict[key] == 'Grand Growth':
                    if value >= 10:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 10 and value >= 0:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < 0 and value >= -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

                elif stage_dict[key] == 'Summer':
                    if value >= 0:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 0 and value >= -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < -10 and value >= -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -20:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'
                
                elif stage_dict[key] == 'Maturity':
                    if value >= 25:
                        inference[key] = 'No Crop Stress'
                        sub_inference[key] = 'NA'
                    elif value < 25 and value >= 0:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Pukkaboeng Stress'
                    elif value < 0 and value >= -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Top Borer Stress'
                    elif value < -10:
                        inference[key] = 'Severe Crop Stress'
                        sub_inference[key] = 'Red Rot Stress'

        else:
            inference[key] = 'Presence of Cloud'
            sub_inference[key] = 'NA'
    
    print("Inference generated")
            
    return inference, sub_inference

### CLIPPING RASTER
def clipping_raster(geojson_data, path):
    with rasterio.open(path) as src:   
        out_image, out_transform = rasterio.mask.mask(src, geojson_data.geometry, crop = True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
    with rasterio.open(path, "w", **out_meta) as dest:
        dest.write(out_image)
    print("Raster clipped")

### ZONAL STATISTICS 
def zonal_stats_calc(geojson_data, path):
    stats = zonal_stats(geojson_data,
                        path,
        			    band = 1,
                        nodata = np.nan,
                        stats = ['mean'], 
                        geojson_out = True)

    stats_mean_dict = {}
    for i in range(len(stats)):
        stats_mean_dict[stats[i]['properties']['PLOT_NO']] = stats[i]['properties']['mean']

    print("Zonal Stats calculated")

    return stats, stats_mean_dict

### EXCEL GENERATION
def excel(stats, inference, sub_inference):
    rows = []
    for i in range(len(stats)):
        rows.append(stats[i]['properties'])
    stats_excel = pd.DataFrame(rows)

    inference_column = pd.DataFrame(list(inference.values()), columns = ['INFERENCE'])
    sub_inference_column = pd.DataFrame(list(sub_inference.values()), columns = ['SUB_INFERENCE'])
    stats_excel['INFERENCE'] = inference_column
    stats_excel['SUB_INFERENCE'] = sub_inference_column

    print("Statistic Excel generated")

    return stats_excel

# Helper functions
def prepare_sentinel_data(geojson_data, input_date):
    bbox, extent = dimensions(geojson_data)
    sentinel_data = sentinel_data_dict(bbox, input_date)
    width = sentinel_data['BAND4_REF'].shape[1]
    height = sentinel_data['BAND4_REF'].shape[0]
    dtype = sentinel_data['BAND4_REF'].dtype
    transform = rasterio.transform.from_bounds(extent[0], extent[1], extent[2], extent[3], width, height)
    return sentinel_data, transform, width, height, dtype,bbox

def process_cy(geojson_data, input_date, crop, sentinel_data, transform, width, height, dtype,bbox):
    cy_path = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\output_data\CY.tiff'
    cy = cy_calc(input_date, crop, sentinel_data, bbox)
    with rasterio.open(cy_path, 'w', driver='GTiff', width=width, height=height, count=1, dtype=dtype, crs=rasterioCRS.from_epsg(4326), transform=transform) as dst:
        dst.write(cy, 1)
    clipping_raster(geojson_data, cy_path)
    print(" in cy")
    print(np.amin(cy), np.amax(cy))
    cy, cy_mean_dict = zonal_stats_calc(geojson_data, cy_path)
    return cy_mean_dict,cy

def process_redsi(geojson_data, sentinel_data, transform, width, height, dtype):
    redsi_path = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\output_data\REDSI.tiff'
    redsi = redsi_calc(sentinel_data)
    with rasterio.open(redsi_path, 'w', driver='GTiff', width=width, height=height, count=1, dtype=dtype, crs=rasterioCRS.from_epsg(4326), transform=transform) as dst:
        dst.write(redsi, 1)

    clipping_raster(geojson_data, redsi_path)
    print(" in redsi")
    min_max = []
    # with rasterio.open(redsi_path) as src:
    #     redsi = src.read(1)
    #     min_max = np.array([round(np.amin(redsi),2), round(np.amax(redsi),2)])
    #     print("min_max",min_max)

    with rasterio.open(redsi_path) as src:
        redsi = src.read()
        redsi_flatten = redsi.flatten()
        redsi_unique = np.unique(redsi_flatten)
        redsi_min = np.amin(redsi)
        redsi_max = np.amax(redsi)
        if np.amin(redsi) == float('-inf'):
            redsi_min = redsi_unique[1]
        if np.amax(redsi) == float('inf'):
            redsi_max = redsi_unique[-2]
        else:
            redsi_min = np.amin(redsi)
            redsi_max = np.amax(redsi)
        min_max = np.array([round(redsi_min,2), round(redsi_max,2)])
        print(redsi_min, redsi_max)

    _, redsi_mean_dict = zonal_stats_calc(geojson_data, redsi_path)

    for key,value in redsi_mean_dict.items():
        if value == float('inf'):
            redsi_mean_dict[key] = 1000

    print(redsi_mean_dict)
    return redsi_mean_dict, redsi,min_max

def process_cloud_mask(geojson_data, sentinel_data, transform, width, height, dtype):
    mask_path = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\output_data\MASK.tiff'
    cloud_prob, cloud_mask = cloud_detector(sentinel_data)
    with rasterio.open(mask_path, 'w', driver='GTiff', width=width, height=height, count=1, dtype=dtype, crs=rasterioCRS.from_epsg(4326), transform=transform) as dst:
        dst.write(cloud_mask[0], 1)
    clipping_raster(geojson_data, mask_path)
    print(" in cloud")
    masks, masks_mean_dict = zonal_stats_calc(geojson_data, mask_path)
    return masks_mean_dict

def generate_geojson(row):
    """
    Generate a GeoJSON polygon feature for a row using Lat_1, Long_1, Lat_2, Long_2, Lat_3, Long_3, Lat_4, Long_4
    and include all other properties from the row.
    """
    # Create a polygon feature using the latitude and longitude columns
    coordinates = [
        [row['Long_1'], row['Lat_1']],
        [row['Long_2'], row['Lat_2']],
        [row['Long_3'], row['Lat_3']],
        [row['Long_4'], row['Lat_4']],
        [row['Long_1'], row['Lat_1']]  # Closing the polygon loop
    ]
    
    # Add all other columns as properties except geometry
    properties = row.to_dict()
    
    # Create the GeoJSON structure
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates]
        },
        "properties": properties
    }
    
    return json.dumps(geojson)


def generate_custom_dataframe(final_df, input_date, parameter):
    """
    This function creates a new dataframe from final_df with the columns:
    Unique_ID, Result (combining REDSI and INFERENCE), Parameter, Date, and geojson_data.
    
    The geojson_data is generated using Lat_1, Long_1, Lat_2, Long_2, Lat_3, Long_3, Lat_4 columns and all
    other properties from final_df are included in the properties section of the GeoJSON.
    
    Args:
    - final_df: DataFrame containing the final data including latitude, longitude, and other properties.
    - input_date: Date string or datetime representing the date for the Date column.
    - parameter: Parameter string to store in the Parameter column.
    
    Returns:
    - temp_df: New DataFrame with the required columns and generated geojson_data.
    """
    
    # Create a new dataframe with the required columns
    temp_df = pd.DataFrame()

    # Create the Unique_ID column (assuming PLOT_NO is used as a unique identifier)
    temp_df['Unique_ID'] = final_df['PLOT_NO']

    # Create the Result column by combining REDSI and INFERENCE
    temp_df['Result'] = final_df.apply(lambda row: f"REDSI: {row['REDSI']}, INFERENCE: {row['INFERENCE']}", axis=1)

    # Create a Parameter column
    temp_df['Parameter'] = parameter

    # Create a Date column using the input_date variable from the function arguments
    temp_df['Date'] = input_date

    # Create the Geojson_data column by applying the generate_geojson function to each row in final_df
    temp_df['geojson_data'] = final_df.apply(generate_geojson, axis=1)

    return temp_df

def generate_final_excel(geojson_data, input_date, crop, cy, redsi, redsi_mean_dict, masks_mean_dict):
    inference, sub_inference = inferencing(crop, input_date, geojson_data, redsi_mean_dict, masks_mean_dict)
    inference_excel = excel(cy, inference, sub_inference)
    inference_excel.rename(columns={'mean': 'CROP_YIELD'}, inplace=True)
    
    redsi_df = pd.DataFrame(list(redsi_mean_dict.values()), columns=['REDSI'])
    final_df = pd.concat([inference_excel, redsi_df], axis=1)
    
    # Reorder columns
    final_df.iloc[:, -3:] = final_df.iloc[:, [-1, -3, -2]].values
    cols = final_df.columns.tolist()
    cols[-3], cols[-2], cols[-1] = cols[-1], cols[-3], cols[-2]
    final_df.columns = cols
    
    temp_df = generate_custom_dataframe(final_df,input_date,"Crop Stress Biotic")

    final_df.to_excel(r'C:/Users/ANUBHAV/OneDrive/Desktop/AGRI_DCM/backend/app/main/output_data/excel.xlsx')
    # Get the minimum and maximum values from the 'REDSI' column
    redsi_min = round(final_df['REDSI'].min(), 2)
    redsi_max = round(final_df['REDSI'].max(), 2)
    
    # Print the min and max values
    print(f"Minimum REDSI: {redsi_min}")
    print(f"Maximum REDSI: {redsi_max}")
    
    redsi_min_max = f"{redsi_min},{redsi_max}"
    print("final excel generated")
    return temp_df,redsi_min_max



