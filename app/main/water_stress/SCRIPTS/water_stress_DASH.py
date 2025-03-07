import os
from dotenv import load_dotenv
import geopandas as gpd
import pandas as pd
import numpy as np
from sentinelhub import CRS as sentinelCRS, BBox, SHConfig
import rasterio
from rasterio.crs import CRS as rasterioCRS
from datetime import datetime, timedelta
import time
import math
import csv
import requests
from scipy.constants import Stefan_Boltzmann
import rasterio.mask
from app.main.water_stress.DL_CLOUD_MASKING.dl_l8s2_uv import utils
from app.main.water_stress.DL_CLOUD_MASKING.dl_l8s2_uv.satreaders import l8image
import shutil
import warnings
from app.main.water_stress.SCRIPTS.landsat8 import *
from app.main.water_stress.SCRIPTS.utils import *
from app.main.helpers.graph_table_helpers import generate_custom_dataframe,save_temp_df_to_db
from app.main.helpers.result_table_helpers import create_result_entry
from flask import current_app as app

warnings.filterwarnings("ignore")


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
    geojson_data_dict = data.get('GeojsonData')
    geojson_data = dict_to_gdf(data.get('GeojsonData'))
    selected_parameter = data.get('selectedParameter')
  
    return geojson_data_dict, geojson_data, input_date_str, input_datetime_string, crop, selected_parameter


### SENTINEL HUB CONFIG SETUP

def setup():
    instance_id = app.config['LANDSAT8_ID']
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


### LAI CALCULATIONS
### LAI --> LEAF AREA INDEX

def LAI_calc(SAVI):
    LAI = (-1) * (np.log((0.69 - SAVI) / 0.59) / 0.91)

    print('LAI calculated')

    return LAI


### SENTINEL DATA DICTINOARY

def sentinel_data_dict(bbox, input_date, extent):
    config, _ = setup()
    B02 = band2_reflectance_call(bbox, input_date, config)
    bqa = bqa_layer_call(bbox, input_date, config)
    NDVI = NDVI_layer_call(bbox, input_date, config)
    SAVI = SAVI_layer_call(bbox, input_date, config)
    LAI = LAI_calc(SAVI)
    B03 = band3_reflectance_call(bbox, input_date, config)
    B04 = band4_reflectance_call(bbox, input_date, config)
    B05 = band5_reflectance_call(bbox, input_date, config)
    B06 = band6_reflectance_call(bbox, input_date, config)
    B07 = band7_reflectance_call(bbox, input_date, config)
    B10_BT = band10_bt_call(bbox, input_date, config)
    dem = dem_layer_call(bbox, input_date, config)

    sentinel_data = {
        'NDVI' : NDVI,
        'SAVI' : SAVI,
        'LAI' : LAI,
        'B02' : B02,
        'B03' : B03, 
        'B04' : B04,
        'B05' : B05,
        'B06' : B06,
        'B07' : B07,
        'B10_BT' : B10_BT,
        'BQA' : bqa,
        'DEM' : dem
    }

    width = sentinel_data['NDVI'].shape[1]
    height = sentinel_data['NDVI'].shape[0]
    dtype = sentinel_data['NDVI'].dtype
    transform = rasterio.transform.from_bounds(extent[0], extent[1], extent[2], extent[3], width, height)

    return sentinel_data, width, height, dtype, transform


### FVC CALCULATIONS

def fvc_calc(sentinel_data):
    NDVI = sentinel_data['NDVI']
    NDVI_max = np.amax(NDVI) #0.9
    NDVI_min = np.amin(NDVI) #0.2
    fvc = ((NDVI - NDVI_min) / (NDVI_max - NDVI_min)) **2

    print("FVC calculated")
    
    return fvc


### EMISSIVITY CALCULATIONS

def es_calc(sentinel_data):
    NDVI = sentinel_data['NDVI']
    B04 = sentinel_data['B04']
    fvc = fvc_calc(sentinel_data)
    Ev = 0.989           ## VEGETATION EMISSIVITY
    Es = 0.977           ## SOIL EMISSIVITY
    F = 0.55             ## GEOMETRIC FORM FACTOR
    ES = []              ## EMISSIVITY
    C = (1 - Es) * (1 - fvc) * F * Ev       ## MODIFICATION IN EMISSIVITY DUE TO CAVITY EFFECT AND MIXED PIXEL SCATTERING
    for i in range(len(NDVI)):
        row = []
        for j in range(len(NDVI[i])):
            if NDVI[i][j] < 0.2:
                row.append(0.979 - (0.046 * B04[i][j]))
            elif NDVI[i][j] <= 0.5 and NDVI[i][j] >= 0.2:
                row.append((Ev * fvc[i][j]) + (Es * (1 - fvc[i][j])) + C[i][j])
            elif NDVI[i][j] > 0.5:
                row.append(Ev + C[i][j])
        ES.append(row)

    ES = np.array(ES)

    print("ES calculated")
    
    return ES


### LST CALCULATIONS

def lst_calc(sentinel_data):
    B10_BT = sentinel_data['B10_BT']
    ES = es_calc(sentinel_data)
    wavelength = 10.895     ## AVERAGE WAVELENGTH OF BAND 10 (um)
    lst_const = 14388       ## (PLANCK CONSTANT * VELOCITY OF LIGHT) / BOLTZMANN CONSTANT
    LST = B10_BT / (1 + (((wavelength * B10_BT) / lst_const) * np.log(ES)))

    print("LST calculated")
    
    return LST


### WEIGHTED ALBEDO CALCULATIONS

def wta_calc(sentinel_data):
    B02 = sentinel_data['B02']
    B03 = sentinel_data['B03']
    B04 = sentinel_data['B04']
    B05 = sentinel_data['B05']
    B06 = sentinel_data['B06']
    B07 = sentinel_data['B07']
    w1 = 0.356 #0.301297899     ## WEIGHT FOR BAND 2 REFLACTANCE (0.356)
    w2 = 0.326 #0.27593465      ## WEIGHT FOR BAND 3 REFLACTANCE (0.326)
    w3 = 0.138 #0.23366257      ## WEIGHT FOR BAND 4 REFLACTANCE (0.138)
    w4 = 0.084 #0.141771812     ## WEIGHT FOR BAND 5 REFLACTANCE (0.084)
    w5 = 0.056 #0.03571262      ## WEIGHT FOR BAND 6 REFLACTANCE (0.056)
    w6 = 0.41 #0.011620449     ## WEIGHT FOR BAND 7 REFLACTANCE (0.41)
    wta = (w1 * B02) + (w2 * B03) + (w3 * B04) + (w4 * B05) + (w5 * B06) + (w6 * B07) 

    print("WTA calculated")
    
    return wta


### ATMOSPHERIC TRANSIMISSIVITY CALCULATIONS

def tsw_calc(sentinel_data):
    dem = sentinel_data['DEM']
    tsw = 0.75 + (2 * 0.00001 * dem)

    print("TSW calculated")
    
    return tsw


### SURFACE ALBEDO CALCULATIONS

def sa_calc(sentinel_data):
    wta = wta_calc(sentinel_data)
    tsw = tsw_calc(sentinel_data)
    pr = 0.03     ## PATH RADIANCE FOR SEBAL
    SA = (wta - pr) / tsw**2

    print("SA calculated")
     
    return SA


### TOKEN ID API CALL

def token_call():
    _, instance_setup = setup()
    url = "https://services.sentinel-hub.com/oauth/token"
    headers = {"content-type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": instance_setup['CLIENT_ID'],
        "client_secret": instance_setup['CLIENT_SECRET']
    }

    response = requests.post(url, headers=headers, data=data)
    # print(response.json())
    token = response.json()
    
    return token['access_token']


### SENTINELHUB METADATA API CALL

def metadata_call(geojson_data_dict, input_datetime_string):
    token = token_call()
    bearer = "Bearer " + token
   
    url = "https://services-uswest2.sentinel-hub.com/api/v1/catalog/1.0.0/search" 
    headers = {   "Content-Type": "application/json",   
                  "Authorization": bearer 
              } 
    data = {  
        "collections": [    
            "landsat-ot-l1"   
        ],   
        "datetime": input_datetime_string, 
        "intersects": geojson_data_dict['features'][0]['geometry'],   
        "limit": 1 
    }  
    response = requests.post(url, headers = headers, json = data)
    response_content = response.json()
    print('Metadata fetched')
    
    return response_content


### SUN EARTH DISTANCE CALCULATIONS

def sun_earth_dist(input_date):
    distance_data = {}
    with open('backend/app/main/water_stress/SCRIPTS/earth_sun.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            distance_data[row['data']] = {'d': row['d']}
    input_date = input_date[5:]
    print("in sun earth dist")
    return distance_data[input_date]['d']


### INCOMING SHORTWAVE RADIATION CALCULATIONS

def rs_calc(geojson_data_dict, input_date, input_datetime_string, sentinel_data):
    sed = sun_earth_dist(input_date)
    sed = float(sed)
    metadata = metadata_call(geojson_data_dict, input_datetime_string)
    sea = metadata["features"][0]["properties"]["view:sun_elevation"]
    sea = math.radians(sea)
    sea = np.sin(sea)
    tsw = tsw_calc(sentinel_data)
    sc = 1367     ## SOLAR CONSTANT (W/m2)
    RS = sc * sea * (1 / (sed * sed)) * tsw

    print("RS calculated")
    
    return RS


### API CALL FOR WEATHER CONDITIONS

def weather_api_call(geojson_data_dict, input_datetime_string, bbox):
    metadata = metadata_call(geojson_data_dict, input_datetime_string)
    unix_time = metadata["features"][0]["properties"]["datetime"]
    date_string = unix_time.replace("T", " ").replace("Z", "")
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    datetime_obj = datetime.strptime(date_string, date_format)
    unixtime = time.mktime(datetime_obj.timetuple())
    unixtime = math.trunc(unixtime)
    _, instance_setup = setup()
    APIKEY = instance_setup['OPENWEATHER_APIKEY']                            
    unixtime = math.trunc(unixtime)
    da = unixtime 
    lat = bbox.middle[1]
    lon = bbox.middle[0]
    url = f'https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={lat}&lon={lon}&dt={da}&appid={APIKEY}'
    response = requests.get(url)
    data = ''
    if response.status_code == 200:
        data = response.json()
    else:
        print('Error:', response.status_code)

    AT = data['data'][0]['temp']
    RH = data['data'][0]['humidity']
    U10 = data['data'][0]['wind_speed']
    
    return AT, RH, U10


### ATMOSPHERIC EMISSIVITY CALCULATIONS

def ea_calc(geojson_data_dict, input_datetime_string, bbox):
    AT, RH, U10 = weather_api_call(geojson_data_dict, input_datetime_string, bbox)
    AT = AT - 273.15
    power = (17.27 * AT) / (237.3 + AT)
    AVP = 6.108 * (RH / 100) * math.exp(power)     ## TETENS EQUATION
    EA = 0.52 + (0.065 * math.sqrt(AVP))           ## BRUNT's EQAUTION
    AT = AT + 273.15

    print("EA calculated")
    
    return EA, AT


### NET RADIATION CALCULATIONS

def r_calc(geojson_data_dict, input_date, input_datetime_string, sentinel_data, bbox):
    LST = lst_calc(sentinel_data)
    EA, AT = ea_calc(geojson_data_dict, input_datetime_string, bbox)
    ES = es_calc(sentinel_data)
    SA = sa_calc(sentinel_data)
    RS = rs_calc(geojson_data_dict, input_date, input_datetime_string, sentinel_data)
    sbc = Stefan_Boltzmann    ## STEFAN BOLTZMANN CONSTANT (W/m2*K4)
    R = ((1 - SA) * RS) + (ES * EA * sbc * (AT ** 4)) - (ES * sbc * (LST ** 4))

    print("R calculated")
    
    return R, LST, SA


### SOIL HEAT FLUX CALCULATIONS

def g_calc(geojson_data_dict, input_date, input_datetime_string, sentinel_data, bbox):
    NDVI = sentinel_data['NDVI']
    R, LST, SA = r_calc(geojson_data_dict, input_date, input_datetime_string, sentinel_data, bbox)
    LST = LST - 273.15
    G = ((R * LST) / SA) * ((0.0038 * SA) + (0.0074 * (SA * SA))) * (1 - (0.98 * NDVI * NDVI * NDVI * NDVI))
    LST = LST + 273.15

    print("G calculated")
    
    return G


### CROP HEIGHT DICTIONARY

def crop_height(crop, sentinel_data):
    crop = crop[0] # its because crop is a list, (just a single variable in the list, return that)
    sugarcane_height = ((-1) * 0.1695 * (sentinel_data['LAI'] * 2.3) * (sentinel_data['LAI'] * 2.3)) + (1.4576 * (sentinel_data['LAI'] * 2.3))
    crop_height_dict = {
        "Sugarcane" : sugarcane_height,
        "Potato" : 0.6,
        "Cotton" : 1.2,
        "Wheat" : 1.0,
        "Corn" : 2.0,
        "Rice" : 1.0,
        "Date Palm" : 4.0
    }

    return crop_height_dict[crop]


### AERODYNAMIC ROUGHNESS LENGTH CALCULATIONS

def zom_calc(crop, sentinel_data):
    height = crop_height(crop, sentinel_data)
    zom = 0.12 * height

    print("ZOM calculated")

    return zom


### FRICTION VELOCITY CALCULATIONS AT WEATHER STATION FOR NEUTRAL CONDITIONS

def fv_calc(geojson_data_dict, input_datetime_string, zom, bbox):
    k = 0.41     ## VON KARMAN CONSTANT
    z_wind = 10     ## HEIGHT AT WHICH WIND SPEED IS MEASURED
    AT, RH, U10 = weather_api_call(geojson_data_dict, input_datetime_string, bbox)
    fv = (k * U10) / np.log((z_wind) / zom)

    print("Friction Velocity calculated")

    return fv


### WIND SPEED CALCULATIONS AT BLENDING HEIGHT

def u_200_calc(zom, fv):
    k = 0.41  ## VON KARMAN CONSTANT
    u_200 = (fv / k) * np.log(200 / zom)

    return u_200


### PIXEL-WISE CALCULATIONS FOR MOMENTUM ROUGHNESS LENGTH

def zom_pixel_calc(sentinel_data):
    NDVI = sentinel_data['NDVI']
    SA = sa_calc(sentinel_data)
    zom_pixel = np.exp((1.096 * NDVI) / (SA - 5.307))

    print("ZOM Pixel calculated")

    return zom_pixel


### PIXEL-WISE FRICTION VELOCITY CALCULATIONS

def fv_pixel_calc(zom_pixel, zom, fv):
    u_200 = u_200_calc(zom, fv)
    k = 0.41     ## VON KARMAN CONSTANT
    fv_pixel = (k * u_200) / np.log(200 / zom_pixel)

    print("Pixel Friction Velocity calculated")

    return fv_pixel


### AERODYNAMIC RESISTANCE CALCULATIONS

def RAH_calc(zom, fv, zom_pixel):
    k = 0.41     ## VON KARMAN CONSTANT
    fv_pixel = fv_pixel_calc(zom_pixel, zom, fv)
    RAH = (np.log((200 - zom_pixel) / zom_pixel)) / (fv_pixel * k)

    print("RAH calculated")

    return RAH


### COLD PIXEL CALCULATIONS

def cold_pixel_calc(LST, LAI, SA, NDVI):
    if np.amax(SA) >= 0.20:
        SA_range = (0.20, 0.3)
    else:
        SA_range = (np.amin(SA), np.mean(SA))
    if np.amax(LAI) >= 4:
        LAI_range = (4, 6)
    else:
        LAI_range = (np.mean(LAI), 6)
    if np.amax(NDVI) >= 0.7:
        NDVI_threshold = 0.7
    else:
        NDVI_threshold = np.amax(NDVI)

    cold_pixel_indices = np.where((LAI >= LAI_range[0]) & (LAI <= LAI_range[1]) &
                                  (NDVI >= NDVI_threshold) & (SA >= SA_range[0]) & (SA <= SA_range[1]))
    
    if len(cold_pixel_indices[0]) == 0:
        print("No Cold Pixel Found")
        return np.amin(LST), 0 , 0
    
    Tcold = np.inf
    cold_pixel = None
    for i in range(len(cold_pixel_indices[0])):
        idx = cold_pixel_indices[0][i]
        idy = cold_pixel_indices[1][i]
        if LST[idx][idy] < Tcold:
            Tcold = LST[idx][idy]
            cold_pixel = [Tcold, idx, idy]

    print("Cold Pixel calculated")
    
    return cold_pixel[0], cold_pixel[1], cold_pixel[2]


### HOT PIXEL CALCULATIONS

def hot_pixel_calc(LST, LAI, SA, NDVI):
    SA_range = (0.3, np.amax(SA))  
    if np.amax(SA) >= 0.3:
        SA_range = (0.3, np.amax(SA)) 
    else:
        SA_range = (np.mean(SA), np.amax(SA))
    if np.amin(LAI) <= 0.4:
        LAI_range = (0, 0.4)  
    else:
        LAI_range = (0, np.amin(LAI))
    if np.amin(NDVI) <= 0.28:
        NDVI_range = (0.1, 0.28)  
    else:
        NDVI_range = (0, np.mean(NDVI))

    hot_pixel_indices = np.where((SA >= SA_range[0]) & (SA <= SA_range[1]) &
                                  (LAI >= LAI_range[0]) & (LAI <= LAI_range[1]) &
                                  (NDVI >= NDVI_range[0]) & (NDVI <= NDVI_range[1]))

    if len(hot_pixel_indices[0]) == 0:
        print("No Hot Pixel Found")
        LST_max_list = []
        for i in range(len(LST)):
            for j in range(len(LST[i])):
                if LST[i][j] == np.amax(LST):
                    LST_max_list.append([LST[i][j], NDVI[i][j], i, j])

        NDVI_min = LST_max_list[0][1]
        hot_pixel = [LST_max_list[0][0], LST_max_list[0][2], LST_max_list[0][3]]
        for i in range(1, len(LST_max_list)):
            if LST_max_list[i][1] < NDVI_min:
                NDVI_min = LST_max_list[i][1]
                hot_pixel = [LST_max_list[i][0], LST_max_list[i][2], LST_max_list[i][3]]
        
        return hot_pixel[0], hot_pixel[1], hot_pixel[2]

    max_LST = -np.inf
    hot_pixel = None
    for i in range(len(hot_pixel_indices[0])):
        idx = hot_pixel_indices[0][i]
        idy = hot_pixel_indices[1][i]
        if LST[idx][idy] > max_LST:
            max_LST = LST[idx][idy]
            hot_pixel = [max_LST, idx, idy]

    print("Hot Pixel calculated")

    return hot_pixel[0], hot_pixel[1], hot_pixel[2]


### MONIN-OBUKHOV LENGTH CALCULATIONS

def L_calc(H, LST, zom_pixel, zom, fv):
    k = 0.41     ## VON KARMAN CONSTANT
    g = 9.81     ## ACCELERATION DUE TO GRAVITY
    rho = 1.3     ## DENSITY OF AIR (kg/m3)
    sh = 1004     ## SPECIFIC HEAT OF AIR (J/kg*K)
    fv_pixel = fv_pixel_calc(zom_pixel, zom, fv)
    L = ((-1) * rho * sh * (fv_pixel **3) * LST) / (k * g * H)
    L[L < -1000] = -1000
          
    print ("Monin-Obukhov Length calculated")

    return L


### STABILITY CORRECTIONS

def stability_corrections(L, zom_pixel):
    z1 = zom_pixel
    z2 = 10 - zom_pixel
    psi_200 = []
    psi_z1 = []
    psi_z2 = []
    for i in range(len(L)):
        row1 = []
        row2 = []
        row3 = []
        for j in range(len(L[i])):
            if L[i][j] < 0:
                x_200 = (1 - ((16 * 200) / L[i][j])) **0.25
                x_z1 = (1 - ((16 * z1[i][j]) / L[i][j])) **0.25
                x_z2 = (1 - ((16 * z2[i][j]) / L[i][j])) **0.25
        
                row1.append((2 * np.log((1 + x_200) / 2)) + (np.log((1 + (x_200 **2)) / 2)) - (2 * math.atan(x_200)) + (0.5 * math.pi))
                row2.append(2 * np.log((1 + (x_z1 **2)) / 2))
                row3.append(2 * np.log((1 + (x_z2 **2)) / 2))
        
            elif L[i][j] > 0:
                row1.append(((-5) * 2) / L[i][j])
                row2.append(((-5) * z1[i][j]) / L[i][j])
                row3.append(((-5) * z2[i][j]) / L[i][j])
        
            else:
                row1.append(0)
                row2.append(0)
                row3.append(0)
        psi_200.append(row1)
        psi_z1.append(row2)
        psi_z2.append(row3)
    psi_200 = np.array(psi_200)
    psi_z1 = np.array(psi_z1)
    psi_z2 = np.array(psi_z2)
        
    print("Stability Corrections calculated")

    return psi_200, psi_z1, psi_z2


### CORRECTED PIXEL-WISE FRICTION VELOCITY CALCULATIONS

def fv_pixel_corrected_calc(psi_200, zom_pixel, zom, fv):
    u_200 = u_200_calc(zom, fv)
    k = 0.41     ## VON KARMAN CONSTANT
    fv_pixel_corrected = (k * u_200) / (np.log(200 / zom_pixel) - psi_200)

    print("Corrected Pixel-Wise Velocity calculated")

    return fv_pixel_corrected


### CORRECTED AERODYNAMIC RESISTANCE CALCULATIONS

def RAH_corrected_calc(psi_z2, psi_z1, fv_pixel_corrected, zom_pixel):
    k = 0.41     ## VON KARMAN CONSTANT
    RAH_corrected = (np.log((200 - zom_pixel) / zom_pixel) - psi_z2 + psi_z1) / (fv_pixel_corrected * k)

    print("Corrected RAH calculated")
    
    return RAH_corrected


### SENSIBLE HEAT FLUX CALCULATIONS

def h_calc(geojson_data_dict, input_datetime_string, crop, sentinel_data, R, G, SA, LST, bbox):
    zom = zom_calc(crop, sentinel_data)
    zom_pixel = zom_pixel_calc(sentinel_data)
    fv = fv_calc(geojson_data_dict, input_datetime_string, zom, bbox)
    RAH = RAH_calc(zom, fv, zom_pixel)
    NDVI = sentinel_data['NDVI']
    LAI = sentinel_data['LAI']
    cold_pixel, cpi, cpj = cold_pixel_calc(LST, LAI, SA, NDVI)
    hot_pixel, hpi, hpj = hot_pixel_calc(LST, LAI, SA, NDVI)
    R_hot = R[hpi][hpj]
    G_hot = G[hpi][hpj]
    RAH_hot = RAH[hpi][hpj]

    dT_cold = 0
    rho = 1.3     ## DENSITY OF AIR (kg/m3)
    sh = 1004     ## SPECIFIC HEAT OF AIR (J/kg*K)
    H_hot = R_hot - G_hot     ## LHF = 0 for Hot Pixel
    dT_hot = (RAH_hot * H_hot) / (rho * sh)
    m = (dT_hot - dT_cold) / (hot_pixel - cold_pixel)
    c = dT_hot - (m * hot_pixel)
    dT = (m * LST) + c
    H = (rho * sh * dT) / RAH

    for i in range(10):
        L = L_calc(H, LST, zom_pixel, zom, fv)
        psi_200, psi_z1, psi_z2 = stability_corrections(L, zom_pixel)
        fv_pixel_corrected = fv_pixel_corrected_calc(psi_200, zom_pixel, zom, fv)
        RAH_corrected = RAH_corrected_calc(psi_z2, psi_z1, fv_pixel_corrected, zom_pixel)
        RAH_hot = RAH_corrected[hpi][hpj]
        dT_hot = (RAH_hot * H_hot) / (rho * sh)
        m = (dT_hot - dT_cold) / (hot_pixel - cold_pixel)
        c = dT_hot - (m * hot_pixel)
        dT = (m * LST) + c
        H = (rho * sh * dT) / RAH_corrected
    
    print("H calculated")
        
    return H


### LATENT HEAT FLUX CALCULATIONS 

def lhf_calc(geojson_data_dict, input_date, input_datetime_string, crop, sentinel_data, bbox):
    R, LST, SA = r_calc(geojson_data_dict, input_date, input_datetime_string, sentinel_data, bbox)
    G = g_calc(geojson_data_dict, input_date, input_datetime_string, sentinel_data, bbox)
    H = h_calc(geojson_data_dict, input_datetime_string, crop, sentinel_data, R, G, SA, LST, bbox)
    LHF = R - G - H

    print("LHF calculated")
    
    return LHF, R, G, H


### GLOBAL HORIZONTAL IRRADIANCE CALCULATIONS

def ghi_calc(input_date, bbox):
    lat = bbox.middle[1]
    lon = bbox.middle[0]
    _, instance_setup = setup()
    API_KEY = instance_setup['OPENWEATHER_APIKEY']
    response = requests.get(f"https://api.openweathermap.org/energy/1.0/solar/data?lat={lat}&lon={lon}&date={input_date}&appid={API_KEY}")
    data = response.json()
    GHI = 0
    for i in data["irradiance"]["daily"]:
        GHI = GHI + i['clear_sky']['ghi']
    GHI = GHI / 24

    print("GHI calculated")
    
    return GHI


### EVAPO-TRANSPORATION CALCULATIONS

def et_calc(geojson_data_dict, geojson_data, input_date, input_datetime_string, crop, sentinel_data, bbox, width , height, transform):
    GHI = ghi_calc(input_date, bbox)
    LHF, R, G, H = lhf_calc(geojson_data_dict, input_date, input_datetime_string, crop, sentinel_data, bbox)
    d_to_s = 86400     ## SECONDS IN A DAY
    lhv = 2.45         ## LATENT HEAT OF VAPORIZATION
    ET = (d_to_s / (lhv * 1000000)) * (LHF / (R - G)) * GHI
    ET[ET < 0] = 0

    et_path = 'backend/app/main/output_data/ET.tiff'
    with rasterio.open(et_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = ET.dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(ET, 1)
    clipping_raster(geojson_data, et_path)
    et_stats, et_mean_dict = zonal_stats_calc(geojson_data, et_path)
    
    return et_stats, R, G, H


### CWSI CALCULATIONS
### CWSI --> CROP WATER STRESS INDEX

def cwsi_calc(R, G, H):
    cwsi = H / (R - G)

    print("CWSI calculated")

    return cwsi


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
     

### SWSI CALCULATIONS
### SWSI --> SOIL WATER STRESS INDEX

def swsi_calc(geojson_data, R, G, H, width , height, transform):
    c1 = 0.46307
    c2 = 1.8094
    cwsi = cwsi_calc(R, G, H)
    for i in range(len(cwsi)):
        for j in range(len(cwsi[i])):
            if cwsi[i][j] < 0:
                cwsi[i][j] = 0
    swsi = c1 * c2 * (cwsi ** (c2 - 1)) * np.exp((-1) * c1 * (cwsi ** c2))

    for i in range(len(swsi)):
        for j in range(len(swsi[i])):
            if swsi[i][j] == np.nan:
                print(swsi[i][j])

    swsi_path = 'backend/app/main/output_data/SWSI.tiff'
    with rasterio.open(swsi_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = swsi.dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(swsi, 1)
    clipping_raster(geojson_data, swsi_path)
    swsi_stats, swsi_mean_dict = zonal_stats_calc(geojson_data, swsi_path)
    swsi_df = pd.DataFrame(list(swsi_mean_dict.values()), columns = ['SWSI'])
    tiff_min_max = get_min_max(swsi)

    print("SWSI calculated")

    return swsi_df, swsi_mean_dict, tiff_min_max


### GROWTH PHASE CALCULATION

def growth_phase_calc(crop, input_date, geojson_data):
    def parse_date(date_str):
        return datetime.strptime(date_str, '%Y-%m-%d')

    input_date = parse_date(input_date)    
    stage_dict = {}
    for i in range(len(geojson_data)):
        growth_phase_dict = {}
        planting_date = geojson_data['PLANT_DAY'][i]
        planting_date = datetime.strptime(planting_date, "%Y-%m-%dT%H:%M:%S").date()

        if crop == 'Sugarcane':
            growth_phase_dict = {
                (f'{planting_date}', f'{planting_date + timedelta(days = 45)}') : 'Germination',
                (f'{planting_date + timedelta(days = 45)}', f'{planting_date + timedelta(days = 120)}') : 'Tillering',
                (f'{planting_date + timedelta(days = 120)}', f'{planting_date + timedelta(days = 250)}') : 'Grand Growth',
                (f'{planting_date + timedelta(days = 250)}', f'{planting_date + timedelta(days = 750)}') : 'Maturity'
            }

        for date_range, stage in growth_phase_dict.items():
            start_date, end_date = map(parse_date, date_range)
            if start_date <= input_date <= end_date:
                stage_dict[geojson_data['FARM_ID'][i]] = stage

    print("Growth Phases calculated")

    return stage_dict


### CREATING METADATA.TXT FILE
def metadata_file(geojson_data_dict, input_datetime_string, sentinel_data, height, width, dtype, transform, user_id):
    metadata = metadata_call(geojson_data_dict, input_datetime_string)
    os.makedirs(f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}", exist_ok = True)
    mtl_file = f"""GROUP = LANDSAT_METADATA_FILE
    GROUP = PROJECTION_ATTRIBUTES
        MAP_PROJECTION = "UTM"
        DATUM = "WGS84"
        ELLIPSOID = "WGS84"
        UTM_ZONE = {int(str(metadata['features'][0]['properties']['proj:epsg'])[-2:])}
        ORIENTATION = "NORTH_UP"
        CORNER_UL_LAT_PRODUCT = {metadata['features'][0]['bbox'][3]}
        CORNER_UL_LON_PRODUCT = {metadata['features'][0]['bbox'][0]}
        CORNER_UR_LAT_PRODUCT = {metadata['features'][0]['bbox'][3]}
        CORNER_UR_LON_PRODUCT = {metadata['features'][0]['bbox'][2]}
        CORNER_LL_LAT_PRODUCT = {metadata['features'][0]['bbox'][1]}
        CORNER_LL_LON_PRODUCT = {metadata['features'][0]['bbox'][0]}
        CORNER_LR_LAT_PRODUCT = {metadata['features'][0]['bbox'][1]}
        CORNER_LR_LON_PRODUCT = {metadata['features'][0]['bbox'][2]}
        CORNER_UL_PROJECTION_X_PRODUCT = {metadata['features'][0]['properties']['proj:bbox'][0]}
        CORNER_UL_PROJECTION_Y_PRODUCT = {metadata['features'][0]['properties']['proj:bbox'][3]}
        CORNER_UR_PROJECTION_X_PRODUCT = {metadata['features'][0]['properties']['proj:bbox'][2]}
        CORNER_UR_PROJECTION_Y_PRODUCT = {metadata['features'][0]['properties']['proj:bbox'][3]}
        CORNER_LL_PROJECTION_X_PRODUCT = {metadata['features'][0]['properties']['proj:bbox'][0]}
        CORNER_LL_PROJECTION_Y_PRODUCT = {metadata['features'][0]['properties']['proj:bbox'][1]}
        CORNER_LR_PROJECTION_X_PRODUCT = {metadata['features'][0]['properties']['proj:bbox'][2]}
        CORNER_LR_PROJECTION_Y_PRODUCT = {metadata['features'][0]['properties']['proj:bbox'][1]}
        REFLECTIVE_LINES = {int(((metadata['features'][0]['properties']['proj:bbox'][3] - metadata['features'][0]['properties']['proj:bbox'][1]) / metadata['features'][0]['properties']['gsd']) + 1)}
        REFLECTIVE_SAMPLES = {int(((metadata['features'][0]['properties']['proj:bbox'][2] - metadata['features'][0]['properties']['proj:bbox'][0]) / metadata['features'][0]['properties']['gsd']) + 1)}
    END_GROUP = PROJECTION_ATTRIBUTES
    GROUP = IMAGE_ATTRIBUTES
        SUN_ELEVATION = {metadata['features'][0]['properties']['view:sun_elevation']}
        DATE_ACQUIRED = {datetime.strptime(metadata['features'][0]['properties']['datetime'][:10], "%Y-%m-%d").date()}
        SCENE_CENTER_TIME = {metadata['features'][0]['properties']['datetime'][11:]}
    END_GROUP = IMAGE_ATTRIBUTES
    GROUP = LEVEL1_RADIOMETRIC_RESCALING
        REFLECTANCE_MULT_BAND_1 = 2.0000E-05
        REFLECTANCE_MULT_BAND_2 = 2.0000E-05
        REFLECTANCE_MULT_BAND_3 = 2.0000E-05
        REFLECTANCE_MULT_BAND_4 = 2.0000E-05
        REFLECTANCE_MULT_BAND_5 = 2.0000E-05
        REFLECTANCE_MULT_BAND_6 = 2.0000E-05
        REFLECTANCE_MULT_BAND_7 = 2.0000E-05
        REFLECTANCE_MULT_BAND_8 = 2.0000E-05
        REFLECTANCE_MULT_BAND_9 = 2.0000E-05
        REFLECTANCE_ADD_BAND_1 = -0.100000
        REFLECTANCE_ADD_BAND_2 = -0.100000
        REFLECTANCE_ADD_BAND_3 = -0.100000
        REFLECTANCE_ADD_BAND_4 = -0.100000
        REFLECTANCE_ADD_BAND_5 = -0.100000
        REFLECTANCE_ADD_BAND_6 = -0.100000
        REFLECTANCE_ADD_BAND_7 = -0.100000
        REFLECTANCE_ADD_BAND_8 = -0.100000
        REFLECTANCE_ADD_BAND_9 = -0.100000
    END_GROUP = LEVEL1_RADIOMETRIC_RESCALING
END_GROUP = LANDSAT_METADATA_FILE
END
"""

    metadata_path = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}/{metadata['features'][0]['id']}_MTL.txt"
    with open(metadata_path, "w") as file:
        file.write(mtl_file)

    b2_path = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}/{metadata['features'][0]['id']}_B2.tif"
    b2 = sentinel_data['B02']
    with rasterio.open(b2_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(b2, 1)

    b3_path = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}/{metadata['features'][0]['id']}_B3.tif"
    b3 = sentinel_data['B03']
    with rasterio.open(b3_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(b3, 1)

    b4_path = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}/{metadata['features'][0]['id']}_B4.tif"
    b4 = sentinel_data['B04']
    with rasterio.open(b4_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(b4, 1)

    b5_path = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}/{metadata['features'][0]['id']}_B5.tif"
    b5 = sentinel_data['B05']
    with rasterio.open(b5_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(b5, 1)

    b6_path = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}/{metadata['features'][0]['id']}_B6.tif"
    b6 = sentinel_data ['B06']
    with rasterio.open(b6_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(b6, 1)

    b7_path = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}/{metadata['features'][0]['id']}_B7.tif"
    b7 = sentinel_data['B07']
    with rasterio.open(b7_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(b7, 1)

    bqa_path = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}/{metadata['features'][0]['id']}_BQA.tif"
    bqa = sentinel_data['BQA']
    with rasterio.open(bqa_path, 'w', driver = 'GTiff', width = width, height = height, count = 1, dtype = dtype, crs = rasterioCRS.from_epsg(4326), transform = transform) as dst:
        dst.write(bqa, 1)

    return metadata


### LANDSAT DL CLOUD MASKING

def cloud_masking(geojson_data_dict, geojson_data, input_datetime_string, sentinel_data, height, width, dtype, transform, user_id):
    metadata = metadata_file(geojson_data_dict, input_datetime_string, sentinel_data, height, width, dtype, transform, user_id)
    utils.select_cuda_device("gpu")
    satname = "L8"
    namemodel = "rgbiswir"
    landsatimage = f"backend/app/main/water_stress/DL_CLOUD_MASKING/{metadata['features'][0]['id']}"
    satobj = l8image.L8Image(landsatimage)
    model = utils.Model(satname = satname, namemodel = namemodel)
    cloud_prob_bin = model.predict(satobj)
    path = os.path.join(satobj.folder, "Mask.tiff")
    utils.save_cloud_mask(satobj, cloud_prob_bin, path)
    clipping_raster(geojson_data, path)
    masks, masks_mean_dict = zonal_stats_calc(geojson_data, path)

    print("Cloud Mask generated")

    return masks, masks_mean_dict, path


### FINAL INFERENCING CALCULATIONS

def inferencing(swsi_mean_dict, masks_mean_dict, crop, input_date, geojson_data):
    stage_dict = growth_phase_calc(crop, input_date, geojson_data)
    inference = {}
    for (key, value) in swsi_mean_dict.items():
        if key in stage_dict.keys():
            if masks_mean_dict[key] == 1:
                if value == None:
                    inference[key] = 'None'
                    continue
                
                if stage_dict[key] == 'Germination':
                    if 0 <= value < 0.15:
                        inference[key] = 'No Water Stress'
                    elif 0.15 <= value < 0.25:
                        inference[key] = 'Medium Water Stress'
                    elif value >= 0.25:
                        inference[key] = 'Severe Water Stress'

                elif stage_dict[key] == 'Tillering':
                    if 0 <= value < 0.2:
                        inference[key] = 'No Water Stress'
                    elif 0.2 <= value < 0.3:
                        inference[key] = 'Medium Water Stress'
                    elif value >= 0.3:
                        inference[key] = 'Severe Water Stress'

                elif stage_dict[key] == 'Grand Growth':
                    if 0 <= value < 0.15:
                        inference[key] = 'No Water Stress'
                    elif 0.15 <= value < 0.25:
                        inference[key] = 'Medium Water Stress'
                    elif value >= 0.25:
                        inference[key] = 'Severe Water Stress'

                elif stage_dict[key] == "Maturity":
                    if 0 <= value < 0.2:
                        inference[key] = 'No Water Stress'
                    elif 0.2 <= value < 0.35:
                        inference[key] = 'Medium Water Stress'
                    elif value >= 0.35:
                        inference[key] = 'Severe Water Stress'

            else:
                inference[key] = 'Presence of Cloud'

        else:
            inference[key] = 'Plantation Cycle Complete'

    print("Inference Calculated")

    return inference


### EXCEL GENERATION

def excel(stats, inference, df):
    rows = []
    for i in range(len(stats)):
        rows.append(stats[i]['properties'])
    stats_excel = pd.DataFrame(rows)

    inference_column = pd.DataFrame(list(inference.values()), columns = ['INFERENCE'])
    stats_excel['INFERENCE'] = inference_column
    stats_excel.rename(columns = {'mean' : 'ET'}, inplace = True)

    final_excel = pd.concat([stats_excel, df], axis = 1)
    final_excel.iloc[:, [-2, -1]] = final_excel.iloc[:, [-1, -2]].values
    cols = final_excel.columns.tolist()
    cols[-1], cols[-2] = cols[-2], cols[-1]
    final_excel.columns = cols
    final_df = pd.DataFrame()
    final_df = pd.concat([final_df, final_excel], axis = 0)
    final_df.reset_index(drop = True, inplace = True)

    print("Statistics Excel generated")

    return final_df


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

def get_data(data):
    project_id = data.get('project_id')
    geojson_data_dict, geojson_data, input_date, input_datetime_string, crop, selected_parameter = input_data(data)
    bbox, extent = dimensions(geojson_data)

    return geojson_data_dict, geojson_data, input_date, input_datetime_string, crop, selected_parameter, bbox, extent, project_id

def generate_tiff(geojson_data_dict,geojson_data, input_date, input_datetime_string, crop, bbox, extent, user_id):
    sentinel_data, width, height, dtype, transform = sentinel_data_dict(bbox, input_date, extent)
    et_stats, R, G, H = et_calc(geojson_data_dict, geojson_data, input_date, input_datetime_string, crop, 
                                sentinel_data, bbox, width, height, transform)
    swsi_df, swsi_mean_dict, tiff_min_max = swsi_calc(geojson_data, R, G, H, width, height, transform)
    _, masks_mean_dict, mask_path = cloud_masking(geojson_data_dict, geojson_data, input_datetime_string, 
                                                  sentinel_data, height, width, dtype, transform, user_id)

    return  swsi_df, swsi_mean_dict, masks_mean_dict, mask_path, et_stats, tiff_min_max

def generate_excel(masks_mean_dict, crop, input_date, geojson_data, et_stats, swsi_df, swsi_mean_dict, mask_path):
    inference = inferencing(swsi_mean_dict, masks_mean_dict, crop, input_date, geojson_data)
    final_df = pd.DataFrame()
    final_df = excel(et_stats, inference, swsi_df) ## final excel
    final_df.to_excel('backend/app/main/output_data/WATER_STRESS.xlsx', index = False)
    shutil.rmtree(os.path.dirname(mask_path))
    
    return final_df


### MAIN FUNCTION

def main(data,user_id):
    print("water-stress-script")
    ## MAIN FUNCTION STEPS  ->
    ## get data (recieving data from frontend and changing format etc)
    ## generate tiff and tiff_min_max
    ## generate excel and final_df
    ## save result(excel,tiff,tiff_min_max,project_id) in result_table and get the result_id
    ## make temporary data_frame from final_df 
    ## save that temporary data_frame in graph table with that result_id
    ## send to frontend

    # get data (recieving data from frontend and changing format etc)
    (geojson_data_dict, geojson_data, input_date, input_datetime_string, 
     crop, selected_parameter, bbox, extent, project_id) = get_data(data)

    # generate tiff, tiff_min_max and related stuff
    (swsi_df, swsi_mean_dict, masks_mean_dict, 
     mask_path, et_stats, tiff_min_max) = generate_tiff(geojson_data_dict,geojson_data, input_date, 
                                                        input_datetime_string, crop,bbox,extent,user_id)
    
    # generate excel,final_df and related stuff
    final_df = generate_excel(masks_mean_dict, crop, input_date, geojson_data,et_stats, swsi_df,swsi_mean_dict,
                   mask_path)
    
    ## save result in result_table and get the result_id
    ## send paths of tiff and excel
    tiff_path = 'backend/app/main/output_data/ET.tiff'
    excel_path = 'backend/app/main/output_data/WATER_STRESS.xlsx'

    ## save result(excel,tiff,tiff_min_max,project_id) in result_table and get the result_id
    result_id = create_result_entry(user_id, tiff_min_max, data.get('date'), data.get('selectedParameter'), data.get('GeojsonData'),
                                    project_id,tiff_path,excel_path)
    
     # make temporary data_frame from final_df for that parameter
    temp_df = generate_custom_dataframe(final_df,data.get("date"),"Water Stress")

    # save that temporary data_frame in graph table
    save_temp_df_to_db(temp_df,result_id,user_id)

    return