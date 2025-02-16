import os
from dotenv import load_dotenv
from sentinelhub import DataCollection, MimeType, WcsRequest, CustomUrlParam,SHConfig
from app.main.water_stress.SCRIPTT.utils import *
import warnings
warnings.filterwarnings("ignore")


### LOADING SENTINEL HUB CONFIGURATION

load_dotenv(r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\.env')
instance_id = os.getenv('SENTINEL_LANDSAT8_ID')
client_id = os.getenv('SENTINEL_CLIENT_ID')
client_secret = os.getenv('SENTINEL_CLIENT_SECRET')


### SENTINEL HUB INSTANCE INFORMATION

from sentinelhub import SHConfig
config = SHConfig()
config_dict = config.__dict__
config.instance_id = instance_id
config.sh_client_id = client_id
config.sh_client_secret = client_secret
    
# print("client id:",config.sh_client_id)
# print("secret:",config.sh_client_secret)
# print("instance_id:",config.instance_id)
config.save()



for key, value in config_dict.items():
    if value is None:
        print(f"{key} is None")
### SENTINELHUB API WCS REQUEST FOR BAND 2 REFLECTANCE LAYER

def band2_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/B02',
        layer = 'B02',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band2_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 2 Reflectance fetched")
    
    return band2_ref


### SENTINELHUB API WCS REQUEST FOR BAND 3 REFLECTANCE LAYER

def band3_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/B03',
        layer = 'B03',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band3_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 3 Reflectance fetched")
    
    return band3_ref


### SENTINELHUB API WCS REQUEST FOR BAND 4 REFLECTANCE LAYER

def band4_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/B04',
        layer = 'B04',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band4_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 4 Reflectance fetched")
    
    return band4_ref


### SENTINELHUB API WCS REQUEST FOR BAND 5 REFLECTANCE LAYER

def band5_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/B05',
        layer = 'B05',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band5_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 5 Reflectance fetched")
    
    return band5_ref


### SENTINELHUB API WCS REQUEST FOR BAND 6 REFLECTANCE LAYER

def band6_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/B06',
        layer = 'B06',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band6_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 6 Reflectance fetched")
    
    return band6_ref


### SENTINELHUB API WCS REQUEST FOR BAND 7 REFLECTANCE LAYER

def band7_reflectance_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/B07',
        layer = 'B07',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band7_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 7 Reflectance fetched")
    
    return band7_ref


### SENTINELHUB API WCS REQUEST FOR BAND 10 BRIGHTNESSS TEMPERATURE LAYER

def band10_bt_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/B10_BT',
        layer = 'B10_BT',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band10_BT = mosaic_tiff(wcs_true_color_img)

    print("Band 10 Brightness Temperature fetched")
    
    return band10_BT


### SENTINELHUB API WCS REQUEST FOR BQA LAYER

def bqa_layer_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/BQA_CLOUD',
        layer = 'BQA_CLOUD',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    bqa = mosaic_tiff(wcs_true_color_img)

    print("BQA fetched")
    
    return bqa


### SENTINELHUB API WCS REQUEST FOR DEM LAYER
### DEM --> DIGITAL ELEVATION MODEL

def dem_layer_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.DEM,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/DEM',
        layer = 'DEM',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    dem = mosaic_tiff(wcs_true_color_img)

    print("DEM fetched")
    
    return dem


### SENTINELHUB API WCS REQUEST FOR NDVI LAYER
### NDVI --> NORMALIZED DIFFERENCE VEGETATION INDEX

def NDVI_layer_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection  = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/NDVI',
        layer = 'NDVI',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()  
    NDVI = mosaic_tiff(wcs_true_color_img)

    print("NDVI fetched")
    
    return NDVI


### SENTINELHUB API WCS REQUEST FOR NDWI LAYER
### NDWI --> NORMALIZED DIFFERENCE WATER INDEX

def NDWI_layer_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection  = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/NDWI',
        layer = 'NDWI',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()  
    NDWI = mosaic_tiff(wcs_true_color_img)

    print("NDWI fetched")
    
    return NDWI


### SENTINELHUB API WCS REQUEST FOR SAVI LAYER
### SAVI --> SOIL ADJUSTED VEGETATION INDEX

def SAVI_layer_call(bbox, input_date):
    wcs_true_color_request = WcsRequest(
        data_collection  = DataCollection.LANDSAT_OT_L1,
        data_folder = '/Users/sid/Documents/STRESS/WATER_STRESS/SENTINEL_PRODUCTS/SAVI',
        layer = 'SAVI',
        bbox = bbox,
        time = input_date,
        resx = '30m',
        resy = '30m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()  
    SAVI = mosaic_tiff(wcs_true_color_img)

    print("SAVI fetched")
    
    return SAVI