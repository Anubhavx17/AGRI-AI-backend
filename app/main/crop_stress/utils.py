import numpy as np
import rasterio
import json
from rasterstats import zonal_stats
import rasterio.mask
import warnings
warnings.filterwarnings("ignore")


### READING GEOJSON DATA

def read_geojson_data(file_path):
    with open(file_path, 'r') as f:
        geojson_data = json.load(f)
        
    return geojson_data


### MOSAIC FUNCTION

def mosaic_tiff(tiffs):
    tiffs = [np.nan_to_num(arr) for arr in tiffs]
    height, width = tiffs[0].shape
    mosaic = np.full(shape = (height, width), fill_value = float('-inf'), dtype = tiffs[0].dtype)
    for tiff in tiffs:
        mosaic = np.maximum(mosaic, tiff)

    print("Mosaic done")

    return mosaic


### CLIPPING RASTER

def clipping_raster(geojson_data, path):
    with rasterio.open(path) as src:   
        out_image, out_transform = rasterio.mask.mask(src, geojson_data.geometry, crop = True, nodata = -9999)
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
                        stats = ['mean'], 
                        geojson_out = True)

    stats_mean_dict = {}
    for i in range(len(stats)):
        stats_mean_dict[stats[i]['properties']['FARM_ID']] = stats[i]['properties']['mean']

    print("Zonal Stats calculated")

    return stats, stats_mean_dict