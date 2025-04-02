from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.data_models.models import ResultModel, GraphModel
from app.data_models.schemas import GraphModelSchema
from app import db
from shapely.geometry import shape
import json
import requests
from app.main.helpers.helpers import get_presigned_url, BUCKET_NAME
from sqlalchemy import text
from sqlalchemy import func


parameter_page_bp = Blueprint('parameter_page_bp', __name__)

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

    

@parameter_page_bp.route('/fetch_graph_details', methods=['POST'])
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

    # Get The inference for that date
    Inference_and_Centroid = GraphModel.query.filter_by(
        unique_farm_id=unique_farm_id,
        user_id=current_user_id,
        selected_date = date,
        selected_parameter= parameter
    ).all()

    geojson = Inference_and_Centroid[0].geojson if Inference_and_Centroid else None
    inference = Inference_and_Centroid[0].inference
    centroid = centroidForZoom(geojson)

    # For the graph get date wise value
    GraphData = GraphModel.query.filter_by(
        unique_farm_id=unique_farm_id,
        user_id=current_user_id,
        selected_parameter= parameter
    ).all()

    # Create schema instance
    graph_schema = GraphModelSchema(many=True)

    # Serialize the data
    serialized_graph_data = graph_schema.dump(GraphData)

    # Print or return serialized data
    date_result_dict = {
    item["selected_date"]: item["result_details"]
    for item in serialized_graph_data
}


    if not Inference_and_Centroid:
        return jsonify({"msg": "No results found for this farm", "results": []}), 404
    
    # Return the serialized results
    return jsonify({"msg": "Results fetched successfully", "centroid":centroid,
                    "Inference":inference,'result':date_result_dict}), 200



def classify_inference(inference_text, parameter):
    """
    Return one of the known categories or None if it doesn't match.
    """
    if parameter == "Water Stress":
        if "No Water Stress" in inference_text:
            return "No Water Stress"
        elif "Medium Water Stress" in inference_text:
            return "Medium Water Stress"
        elif "Severe Water Stress" in inference_text:
            return "Severe Water Stress"
        else:
            # No match => don't categorize
            return None

    elif parameter == "Crop Stress":
        if "No Crop Stress" in inference_text:
            return "No Crop Stress"
        elif "Severe Crop Stress" in inference_text:
            return "Severe Crop Stress"
        else:
            return None

    else:
        # e.g. "Crop Growth"
        if "Ideal Crop Growth" in inference_text:
            return "Ideal Crop Growth"
        elif "Average Crop Growth" in inference_text:
            return "Average Crop Growth"
        elif "Poor Crop Growth" in inference_text:
            return "Poor Crop Growth"
        else:
            return None

@parameter_page_bp.route('/get_stresswise_farms', methods=['POST'])
@jwt_required()
def fetch_inference_data():
    data = request.get_json()
    user_id = get_jwt_identity()
    parameter = data.get('selectedParam')
    selected_date = data.get('currentDate')

    # Query only for this user, parameter, date
    print('parameter',parameter, 'date',selected_date)
    query = GraphModel.query.filter_by(user_id=user_id)

    if parameter:
        query = query.filter_by(selected_parameter=parameter)
    if selected_date:
        query = query.filter_by(selected_date=selected_date)

    graph_data = query.all()

    categorized_results = {}

    for row in graph_data:
        category = classify_inference(row.inference, parameter)
        
        # Skip if category is None (unknown)
        if not category:
            continue

        # Initialize category list if not present
        if category not in categorized_results:
            categorized_results[category] = []
            
        categorized_results[category].append({
            "geojson": row.geojson,
            "unique_farm_id": row.unique_farm_id,
            "result_details": row.result_details
        })

    for category, objects in categorized_results.items():
            print(f"Inference Type: {category}, Number of Objects: {len(objects)}")

    return jsonify(categorized_results), 200


@parameter_page_bp.route('/send_all_result_data', methods = ['POST'])
@jwt_required()
def send_all_data():
    data = request.get_json()
    user_id = get_jwt_identity()
    project_id = data.get('project_id')

    results = ResultModel.query.filter_by(user_id=user_id, project_id=project_id).all()
    clients = []

    for result in results:
        presigned_tiff_url = get_presigned_url(BUCKET_NAME, f"{result.id}.tiff")
        presigned_excel_url = get_presigned_url(BUCKET_NAME, f"{result.id}.xlsx")
        tile_url = (
        f"http://127.0.0.1:{result.port_id}/tiles/{{z}}/{{x}}/{{y}}.png"
        f"?nodata=-9999&colormap_name=viridis&rescale={result.tiff_min_max}"
    )
        clients.append({
            'selected_date': result.selected_date,
            'tiff_url': presigned_tiff_url,
            'tiff_min_max': result.tiff_min_max,
            'excel_url': presigned_excel_url,
            'selected_parameter': result.selected_parameter,
            "geojson": result.geojson,
            "port_id":result.port_id,
             "tile_url": tile_url,
             "active":result.client_active
        })

    try:
        response = requests.post("http://127.0.0.1:5001/initialize_clients", json=clients, timeout=50)
        response.raise_for_status()
        print("Clients initialized successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error initializing clients: {e}")
        return jsonify({"error": "Failed to initialize clients"}), 500

    return jsonify(clients)



@parameter_page_bp.route('/close_clients', methods = ['POST'])
@jwt_required()
def close_clients():
    clients = request.get_json()

    try:
        response = requests.post("http://127.0.0.1:5001/shutdown_clients", json=clients, timeout=50)
        response.raise_for_status()
        print("Clients closed successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error closing clients: {e}")
        return jsonify({"error": "Failed to close clients"}), 500

    return jsonify(clients)

