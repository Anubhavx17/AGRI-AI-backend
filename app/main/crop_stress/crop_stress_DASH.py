import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sentinelhub import CRS as sentinelCRS, BBox, SHConfig
from s2cloudless import S2PixelCloudDetector
import rasterio
from rasterio.crs import CRS as rasterioCRS
import requests
import rasterio.mask
from backend.app.main.crop_stress.sentinel2 import *
from backend.app.main.crop_stress.utils import *
import warnings
from app.main.helpers.graph_table_helpers import generate_custom_dataframe,save_temp_df_to_db
from app.main.helpers.result_table_helpers import create_result_entry
from flask import current_app as app

warnings.filterwarnings("ignore")


### DICTIONARY TO GEODATAFRAME CONVERSION

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


### GEODATAFRAME TO DICTIONARY CONVERSION

def gdf_to_dict(gdf):
    """
    Converts a GeoDataFrame back into a GeoJSON-like dictionary.
    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {k: v for k, v in feature.items() if k != "geometry"},
                "geometry": feature["geometry"].__geo_interface__
            }
            for feature in gdf.to_dict("records")
        ]
    }


### INPUT DATA

def input_data(data):
    input_date = data['date']
    input_date_obj = None
    input_date_obj = datetime.strptime(input_date, "%d/%m/%Y")

    input_date_str = input_date_obj.date().isoformat()  # Extract date in ISO format (YYYY-MM-DD)

    # Create the datetime string for the full day
    input_datetime_string = f"{input_date_str}T00:00:00Z/{input_date_str}T23:59:59Z"
    print(f"Input datetime string: {input_datetime_string}")

    crop = data['selectedCrop']
    geojson_data = dict_to_gdf(data.get('GeojsonData'))
    selected_parameter = data.get('selectedParameter')
  
    return geojson_data, input_date_str, crop, selected_parameter


### SENTINEL HUB CONFIG SETUP

def setup():
    instance_id = app.config['SENTINEL2_ID']
    client_id = app.config['CLIENT_ID']
    client_secret = app.config['CLIENT_SECRET']
    openweather_apikey = app.config['OPENWEATHER_API_KEY']
    instance_setup = {
        'INSTANCE_ID' : instance_id, 
        'CLIENT_ID' : client_id,
        'CLIENT_SECRET' : client_secret,
        'OPENWEATHER_APIKEY' : openweather_apikey
    }

    config = SHConfig()
    print("SentinelHub Configuration:", config)
    config.instance_id = instance_id
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.save()

    print("Config file created")

    return config, instance_setup


### DEFINING BOUNDING BOX

def dimensions(geojson_data):
    minx, miny, maxx, maxy = geojson_data.total_bounds
    extent = [minx, miny, maxx, maxy]
    bbox = BBox(bbox = extent, crs = sentinelCRS.WGS84)

    print("Bounding Box created")
    
    return bbox, extent


### SENTINEL DATA DICTINOARY

def sentinel_data_dict(bbox, input_date, extent):
    config, _ = setup()
    B01 = band1_reflectance_call(bbox, input_date, config)
    B02 = band2_reflectance_call(bbox, input_date, config)
    B04 = band4_reflectance_call(bbox, input_date, config)
    B05 = band5_reflectance_call(bbox, input_date, config)
    B07 = band7_reflectance_call(bbox, input_date, config)
    B08 = band8_reflectance_call(bbox, input_date, config)
    B8A = band8a_reflectance_call(bbox, input_date, config)
    B09 = band9_reflectance_call(bbox, input_date, config)
    B10 = band10_reflectance_call(bbox, input_date, config)
    B11 = band11_reflectance_call(bbox, input_date, config)
    B12 = band12_reflectance_call(bbox, input_date, config)
    NDVI = ndvi_layer_call(bbox, input_date, config)
    LAI = lai_layer_call(bbox, input_date, config)
    FAPAR = fapar_layer_call(bbox, input_date, config)
    LSWI = lswi_layer_call(bbox, input_date, config)

    sentinel_data = {
        'B01' : B01,
        'B02' : B02,
        'B04' : B04,
        'B05' : B05,
        'B07' : B07,
        'B08' : B08,
        'B8A' : B8A,
        'B09' : B09,
        'B10' : B10,
        'B11' : B11,
        'B12' : B12,
        'NDVI' : NDVI,
        'LAI' : LAI,
        'FAPAR' : FAPAR,
        'LSWI' : LSWI
    }

    width = sentinel_data['B04'].shape[1]
    height = sentinel_data['B04'].shape[0]
    dtype = sentinel_data['B04'].dtype
    transform = rasterio.transform.from_bounds(extent[0], extent[1], extent[2], extent[3], width, height)

    return sentinel_data, width, height, dtype, transform


### CLOUD MASK CALCULATIONS

def cloud_detector(geojson_data, sentinel_data, width, height, transform):
    band1_ref = sentinel_data['B01']
    band2_ref = sentinel_data['B02']
    band4_ref = sentinel_data['B04']
    band5_ref = sentinel_data['B05']
    band8_ref = sentinel_data['B08']
    band8a_ref = sentinel_data['B8A']
    band9_ref = sentinel_data['B09']
    band10_ref = sentinel_data['B10']
    band11_ref = sentinel_data['B11']
    band12_ref = sentinel_data['B12']

    bands = [band1_ref, band2_ref, band4_ref, band5_ref, band8_ref, 
             band8a_ref, band9_ref, band10_ref, band11_ref, band12_ref]
    layer_stack = np.stack(bands, axis = -1)

    cloud_detector = S2PixelCloudDetector(threshold = 0.4, average_over = 4, dilation_size = 2, all_bands = False)
    cloud_prob = cloud_detector.get_cloud_probability_maps(layer_stack[np.newaxis, ...])
    cloud_mask = cloud_detector.get_cloud_masks(layer_stack[np.newaxis, ...])

    mask_path = 'backend/app/main/output_data/STRESS_MASK.tiff'
    with rasterio.open(mask_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = cloud_mask[0].dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(cloud_mask[0], 1)
    clipping_raster(geojson_data, mask_path)
    _, mask_mean_dict = zonal_stats_calc(geojson_data, mask_path)

    print("Cloud Probabilities and Mask generated")

    return mask_mean_dict


### TIFF MIN MAX CALCULATIONS

def get_min_max(tiff):
        tiff_flatten = tiff.flatten()
        tiff_unique = np.unique(tiff_flatten)
        print(tiff_unique[:10])
        print(tiff_unique[-10:])
        tiff_min = None
        tiff_max = None
        if np.amin(tiff) == float('-inf'):
            tiff_min = tiff_unique[1]
        if np.amax(tiff) == float('inf'):
            tiff_max = tiff_unique[-2]
        else:
            tiff_min = np.nanmin(tiff)
            tiff_max = np.nanmax(tiff)

        print(tiff_min, tiff_max)

        print("TIFF Min Max calculated")

        return np.array([round(tiff_min,2), round(tiff_max,2)])


### REDSI CALCULATIONS

def redsi_calc(geojson_data, sentinel_data, width, height, dtype, transform):
    band4_ref = sentinel_data['BAND4_REF']
    band5_ref = sentinel_data['BAND5_REF']
    band7_ref = sentinel_data['BAND7_REF']
    # 665nm --> wavelength of band4 reflectance
    # 705nm --> wavelength of band5 reflectance
    # 783nm --> wavelength of band7 reflectance
    redsi = (((705 - 665) * (band7_ref - band4_ref)) - ((783 - 665) * (band5_ref - band4_ref))) / (2 * band4_ref)

    redsi_path = 'backend/app/main/output_data/REDSI.tiff'
    with rasterio.open(redsi_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(redsi, 1)
    clipping_raster(geojson_data, redsi_path)
    redsi_stats, redsi_mean_dict = zonal_stats_calc(geojson_data, redsi_path)
    tiff_min_max = get_min_max(redsi)

    print('REDSI Calculated')

    return redsi_stats, redsi_mean_dict, tiff_min_max


### RATOON DICTIONARY CALCULATIONS

def ratoon_dict_calc(geojson_data):
    ratoon_dict = {}
    for i in range(len(geojson_data)):
        ratoon_dict[geojson_data['FARM_ID'][i]] = geojson_data['TYPE'][i]

    print("Ratoon dictionary generated")

    return ratoon_dict


### STRESS GROWTH PHASE CALCULATION

def stress_phase_calc(crop, input_date, geojson_data):
    def parse_date(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d')

    input_date = parse_date(input_date)  
    stress_stage_dict = {}
    for i in range(len(geojson_data)):
        phase_dict = {}
        planting_date = geojson_data['PLANT_DAY'][i].date()
        if crop == 'Sugarcane':
            if '2023-09-01' <= planting_date.strftime("%Y-%m-%d") <= '2023-12-31':    ## AUTUMN PLANTS
                phase_dict = {
                    (f'{planting_date}', f'{planting_date + timedelta(days = 50)}') : 'Germination',
                    (f'{planting_date + timedelta(days = 50)}', f'{planting_date + timedelta(days = 120)}') : 'Tillering',
                    (f'{planting_date + timedelta(days = 120)}', f'{planting_date + timedelta(days = 175)}') : 'Grand Growth',
                    (f'{planting_date + timedelta(days = 175)}', f'{planting_date + timedelta(days = 250)}') : 'Summer',
                    (f'{planting_date + timedelta(days = 250)}', f'{planting_date + timedelta(days = 700)}') : 'Maturity'
                }
            elif '2024-01-01' <= planting_date.strftime("%Y-%m-%d") <= '2024-05-31':    ## SPRING PLANTS
                phase_dict = {
                    (f'{planting_date}', f'{planting_date + timedelta(days = 45)}') : 'Germination',
                    (f'{planting_date + timedelta(days = 45)}', f'{planting_date + timedelta(days = 115)}') : 'Tillering',
                    (f'{planting_date + timedelta(days = 115)}', f'{planting_date + timedelta(days = 185)}') : 'Monsoon',
                    (f'{planting_date + timedelta(days = 185)}', f'{planting_date + timedelta(days = 218)}') : 'Grand Growth',
                    (f'{planting_date + timedelta(days = 218)}', f'{planting_date + timedelta(days = 250)}') : 'Later Grand Growth',
                    (f'{planting_date + timedelta(days = 250)}', f'{planting_date + timedelta(days = 700)}') : 'Maturity'
                }
    
        for date_range, stage in phase_dict.items():
            start_date, end_date = map(parse_date, date_range)
            if start_date <= input_date <= end_date:
                stress_stage_dict[geojson_data['FARM_ID'][i]] = stage

    print("Stress Growth Stage dictionary created")

    return stress_stage_dict


### STRESS INFERENCING CALCULATIONS

def inferencing(crop, input_date, geojson_data, redsi_stats, redsi_mean_dict, masks_mean_dict):
    ratoon_dict = ratoon_dict_calc(geojson_data)
    stress_stage_dict = stress_phase_calc(crop, input_date, geojson_data)
    inference = {}
    sub_inference = {}
    for key, value in redsi_mean_dict.items():
        if key in stress_stage_dict.keys():
            if masks_mean_dict[key] == 0:
                if ratoon_dict[key] == 'Spring' or ratoon_dict[key] == 'Spring_Ratoon':
                    if stress_stage_dict[key] == 'Germination':
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

                    elif stress_stage_dict[key] == 'Tillering':
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

                    elif stress_stage_dict[key] == 'Monsoon':
                        if value >= 40:
                            inference[key] = 'No Crop Stress'
                            sub_inference[key] = 'NA'
                        elif value < 40 and value >= 30:
                            inference[key] = 'Severe Crop Stress'
                            sub_inference[key] = 'Pukkaboeng Stress'
                        elif value < 30 and value >= 20:
                            inference[key] = 'Severe Crop Stress'
                            sub_inference[key] = 'Top Borer Stress'
                        elif value < 20:
                            inference[key] = 'Severe Crop Stress'
                            sub_inference[key] = 'Red Rot Stress'

                    elif stress_stage_dict[key] == 'Grand Growth':
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

                    elif stress_stage_dict[key] == 'Later Grand Growth':
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

                    elif stress_stage_dict[key] == 'Maturity':
                        if value >= -10:
                            inference[key] = 'No Crop Stress'
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
                    
                elif ratoon_dict[key] == 'Autumn' or ratoon_dict[key] == 'Autumn_Ratoon':
                    if stress_stage_dict[key] == 'Germination':
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

                    elif stress_stage_dict[key] == 'Tillering':
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

                    elif  stress_stage_dict[key] == 'Grand Growth':
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

                    elif stress_stage_dict[key] == 'Summer':
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
                    
                    elif stress_stage_dict[key] == 'Maturity':
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
        
        else:
            inference[key] = 'Plantation Cycle Complete'
            sub_inference[key] = 'NA'
    
    print("Stress Inferencing generated")
            
    return inference, sub_inference


### EXCEL GENERATION

def excel(stats, inference, sub_inference):
    rows = []
    for i in range(len(stats)):
        rows.append(stats[i]['properties'])
    stats_excel = pd.DataFrame(rows)

    inference_column = pd.DataFrame(list(inference.values()), columns = ['INFERENCE'])
    stats_excel['INFERENCE'] = inference_column
    if sub_inference != None:
        sub_inference_column = pd.DataFrame(list(sub_inference.values()), columns = ['SUB_INFERENCE'])
        stats_excel['SUB_INFERENCE'] = sub_inference_column

    print("Statistic Excel generated")

    return stats_excel


### FETCHING INPUT DATA

def get_data(data):
    project_id = data.get('project_id')
    geojson_data, input_date, crop, selected_parameter = input_data(data)
    bbox, extent = dimensions(geojson_data)

    print("Input Data fetched")

    return geojson_data, input_date, crop, selected_parameter, bbox, extent, project_id


### TIFF GENERATION

def generate_tiff(geojson_data, input_date, bbox, extent, user_id):
    sentinel_data, width, height, dtype, transform = sentinel_data_dict(bbox, input_date, extent)
    redsi_stats, redsi_mean_dict, tiff_min_max = redsi_calc(geojson_data, sentinel_data, width, height, dtype, transform)
    masks_mean_dict = cloud_detector(geojson_data, sentinel_data, width, height, transform)

    print("Tiff generated")

    return  redsi_stats, redsi_mean_dict, masks_mean_dict, tiff_min_max


### FINAL EXCEL GENERATION

def generate_excel(crop, input_date, geojson_data, redsi_stats, redsi_mean_dict, masks_mean_dict):
    inference, sub_inference = inferencing(crop, input_date, geojson_data, redsi_stats, redsi_mean_dict, masks_mean_dict)
    final_df = pd.DataFrame()
    final_df = excel(redsi_stats, inference, sub_inference) ## final excel
    final_df.to_excel('backend/app/main/output_data/CROP_STRESS.xlsx', index = False)

    print("Excel generated")
    
    return final_df


### MAIN FUNCTION

def main (data,user_id):
    print("crop-growth-script")
    ## MAIN FUNCTION STEPS  ->
    ## get data (recieving data from frontend and changing format etc)
    ## generate tiff and tiff_min_max
    ## generate excel and final_df
    ## save result(excel,tiff,tiff_min_max,project_id) in result_table and get the result_id
    ## make temporary data_frame from final_df 
    ## save that temporary data_frame in graph table with that result_id
    ## send to frontend

    # get data (recieving data from frontend and changing format etc)
    (geojson_data, input_date, crop, selected_parameter, bbox, extent, project_id) = get_data(data)

    # generate tiff, tiff_min_max and related stuff
    (redsi_stats, redsi_mean_dict, masks_mean_dict, tiff_min_max) = generate_tiff(geojson_data, input_date, 
                                                                                  bbox, extent, user_id)
    
    # generate excel,final_df and related stuff
    final_df = generate_excel(crop, input_date, geojson_data, redsi_stats, redsi_mean_dict, masks_mean_dict)

    ## save result in result_table and get the result_id
    ## send paths of tiff and excel
    tiff_path = 'backend/app/main/output_data/REDSI.tiff'
    excel_path = 'backend/app/main/output_data/CROP_STRESS.xlsx'

    ## save result(excel,tiff,tiff_min_max,project_id) in result_table and get the result_id
    result_id = create_result_entry(user_id, tiff_min_max, data.get('date'), data.get('selectedParameter'), data.get('GeojsonData'),
                                    project_id,tiff_path,excel_path)
    
     # make temporary data_frame from final_df for that parameter
    temp_df = generate_custom_dataframe(final_df,data.get("date"),"Crop Growth")

    # save that temporary data_frame in graph table
    save_temp_df_to_db(temp_df,result_id,user_id)

    return