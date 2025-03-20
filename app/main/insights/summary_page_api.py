from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.data_models.models import ResultModel, GraphModel
from app.data_models.schemas import CropStressGraphModelSchema
from app import db
from shapely.geometry import shape
import json

from sqlalchemy import text
from sqlalchemy import func

summary_page_bp = Blueprint('summary_page_bp', __name__)


@summary_page_bp.route('/get_dates_analysed', methods=['POST'])
@jwt_required()
def get_dates_analysed():
    data = request.get_json()
    user_id = get_jwt_identity()
    project_id = data.get('project_id')


     # 1. Just get the count (single integer)
    project_count = db.session.query(
        func.count(ResultModel.id)
    ).filter(
        ResultModel.user_id == user_id,
        ResultModel.project_id == project_id
    ).scalar()

    # 2. Return it as JSON
    print(project_count)
    return jsonify({
        "project_count": project_count
    })


def centroidForZoom(geojson_data):
    # Ensure geojson_data is a dictionary, not a string
    if isinstance(geojson_data, str):
        geojson_data = json.loads(geojson_data)

    # Convert coordinates from strings to floats (if necessary)
    geojson_data["geometry"]["coordinates"] = [
        [[float(lon), float(lat)] for lon, lat in polygon]
        for polygon in geojson_data["geometry"]["coordinates"]
    ]

    # Convert GeoJSON to a Shapely Polygon
    polygon = shape(geojson_data["geometry"])
    
    # Get the centroid
    centroid = polygon.centroid

    # Return in required format
    return {"latitude": centroid.y, "longitude": centroid.x}


@summary_page_bp.route('/get_inference_data', methods=['POST'])
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
    

@summary_page_bp.route('/fetch_graph_details', methods=['POST'])
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
    date = data.get('date')
    # Get the unique farm ID from the request
    parameter = data.get('parameter')
    print(data,"ok", current_user_id)
    unique_farm_id = data.get('unique_farm_id')

    # Query the database for the requested farm and the current user
    results = GraphModel.query.filter_by(
        unique_farm_id=unique_farm_id,
        user_id=current_user_id,
        selected_date = date,
        selected_parameter= parameter
    ).all()

   
    geojson = results[0].geojson if results else None
    inference = results[0].inference
    results_details = results[0].result_details
    print(results_details)
    centroid = centroidForZoom(geojson)

    if not results:
        return jsonify({"msg": "No results found for this farm", "results": []}), 404
    
    # Return the serialized results
    return jsonify({"msg": "Results fetched successfully", "centroid":centroid,
                    "Inference":inference}), 200