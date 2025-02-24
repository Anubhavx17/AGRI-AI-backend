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
from app.data_models.models import UserModel, GraphModel,ResultModel
from app.data_models.schemas import CropStressGraphModelSchema
from app.main.helpers.result_table_helpers import result_already_exists
from app.main.helpers.helpers import initialize_user_folder
import pandas as pd
from app.main.water_stress.SCRIPTT.water_stress_DASH import main

water_stress_bp = Blueprint('water_stress_api', __name__)
crop_stress_graph_schema = CropStressGraphModelSchema(many=True)

@water_stress_bp.route('/water_stress_api', methods=['POST'])
@jwt_required()
def run_model():
        data = request.get_json()
        # print(data)
        user_id = get_jwt_identity()
        # initialize_user_folder(user_id)
        # Call the main function with parameters
        # existing_result_response = result_already_exists(data.get('date'), selected_parameter, project_id, user_id)
        result = main(data,user_id)
        return jsonify({
            "status": "Success",
            "autumn_excel": result["autumn_excel"],
            "spring_excel": result["spring_excel"]
        }), 200


    

