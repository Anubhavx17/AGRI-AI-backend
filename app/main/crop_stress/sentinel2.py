from sentinelhub import DataCollection, MimeType, WcsRequest, CustomUrlParam
from backend.app.main.crop_stress.utils import *
import warnings

warnings.filterwarnings("ignore")


### SENTINELHUB API WCS REQUEST FOR BAND 1 REFLECTANCE LAYER

def band1_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band1_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 1 Reflectance fetched")
    
    return band1_ref


### SENTINELHUB API WCS REQUEST FOR BAND 2 REFLECTANCE LAYER

def band2_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band2_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 2 Reflectance fetched")
    
    return band2_ref


### SENTINELHUB API WCS REQUEST FOR BAND 3 REFLECTANCE LAYER

def band3_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band3_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 3 Reflectance fetched")
    
    return band3_ref


### SENTINELHUB API WCS REQUEST FOR BAND 4 REFLECTANCE LAYER

def band4_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band4_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 4 Reflectance fetched")
    
    return band4_ref


### SENTINELHUB API WCS REQUEST FOR BAND 5 REFLECTANCE LAYER

def band5_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band5_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 5 Reflectance fetched")
    
    return band5_ref


### SENTINELHUB API WCS REQUEST FOR BAND 6 REFLECTANCE LAYER

def band6_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band6_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 6 Reflectance fetched")
    
    return band6_ref


### SENTINELHUB API WCS REQUEST FOR BAND 7 REFLECTANCE LAYER

def band7_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band7_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 7 Reflectance fetched")
    
    return band7_ref


### SENTINELHUB API WCS REQUEST FOR BAND 8 REFLECTANCE LAYER

def band8_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band8_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 8 Reflectance fetched")
    
    return band8_ref


### SENTINELHUB API WCS REQUEST FOR BAND 8A REFLECTANCE LAYER

def band8a_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band8a_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 8A Reflectance fetched")
    
    return band8a_ref


### SENTINELHUB API WCS REQUEST FOR BAND 9 REFLECTANCE LAYER

def band9_reflectance_call(bbox, input_date, config):
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
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band9_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 9 Reflectance fetched")
    
    return band9_ref


### SENTINELHUB API WCS REQUEST FOR BAND 10 REFLECTANCE LAYER

def band10_reflectance_call(bbox, input_date, config):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L1C,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B10',
        layer = 'B10',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band10_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 10 Reflectance fetched")
    
    return band10_ref


### SENTINELHUB API WCS REQUEST FOR BAND 11 REFLECTANCE LAYER

def band11_reflectance_call(bbox, input_date, config):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B11',
        layer = 'B11',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band11_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 11 Reflectance fetched")
    
    return band11_ref


### SENTINELHUB API WCS REQUEST FOR BAND 12 REFLECTANCE LAYER

def band12_reflectance_call(bbox, input_date, config):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_STRESS/SENTINEL_PRODUCTS/B12',
        layer = 'B12',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    band12_ref = mosaic_tiff(wcs_true_color_img)

    print("Band 12 Reflectance fetched")
    
    return band12_ref


### SENTINELHUB API WCS REQUEST FOR NDVI LAYER
### NDVI --> NORMALIZED DIFFERENCE VEGETATION INDEX

def ndvi_layer_call(bbox, input_date, config):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_GROWTH/SENTINEL_PRODUCTS/NDVI',
        layer = 'NDVI',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    ndvi = mosaic_tiff(wcs_true_color_img)

    print("NDVI fetched")
    
    return ndvi


### SENTINELHUB API WCS REQUEST FOR LAI LAYER
### LAI --> LEAF AREA INDEX

def lai_layer_call(bbox, input_date, config):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_GROWTH/SENTINEL_PRODUCTS/LAI',
        layer = 'LAI',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    lai = mosaic_tiff(wcs_true_color_img)

    print("LAI fetched")
    
    return lai


### SENTINELHUB API WCS REQUEST FOR FAPAR LAYER
### FAPAR --> FRACTION OF ABSORBED PHOTOSYNTHETICALLY ACTIVE RADIATION

def fapar_layer_call(bbox, input_date, config):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_GROWTH/SENTINEL_PRODUCTS/FAPAR',
        layer = 'FAPAR',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    FAPAR = mosaic_tiff(wcs_true_color_img)

    print("FAPAR fetched")
    
    return FAPAR


### SENTINELHUB API WCS REQUEST FOR LSWI LAYER
### LSWI --> LAND SURFACE WATER INDEX

def lswi_layer_call(bbox, input_date, config):
    wcs_true_color_request = WcsRequest(
        data_collection = DataCollection.SENTINEL2_L2A,
        data_folder = '/Users/sid/Documents/STRESS/CROP_GROWTH/SENTINEL_PRODUCTS/LSWI',
        layer = 'LSWI',
        bbox = bbox,
        time = input_date,
        resx = '10m',
        resy = '10m',
        image_format = MimeType.TIFF,
        custom_url_params = {
            CustomUrlParam.SHOWLOGO: False
        },
        config = config
    )
    wcs_true_color_img = wcs_true_color_request.get_data()
    LSWI = mosaic_tiff(wcs_true_color_img)

    print("LSWI fetched")
    
    return LSWI