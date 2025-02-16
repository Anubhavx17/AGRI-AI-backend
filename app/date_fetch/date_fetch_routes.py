from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
import requests
import json
import warnings
from app.save_project.project_routes import get_centroid_of_geojson
from flask import current_app as app

warnings.filterwarnings("ignore")

BUCKET_NAME = "dcmbucket"

# Create a Blueprint for the date fetching routes
date_fetch_bp = Blueprint('date_fetch', __name__)

def filter_geojson_by_polygon_ids(geojsonData, Polygon_IDS):
    # Initialize a new dictionary to hold the filtered features
    filtered_geojsonData = {
        "type": "FeatureCollection",
        "features": []
    }
    # Iterate through the features and add only those with a Polygon_ID in Polygon_IDS
    for feature in geojsonData.get('features', []):
        if feature.get('properties', {}).get('Polygon_ID') in Polygon_IDS:
            filtered_geojsonData['features'].append(feature)
    return filtered_geojsonData


def save_geojson_to_file(geojsonList, filename='okjs.json'):
    # Save the updated geojsonList as a JSON file
    with open(filename, 'w') as file:
        json.dump(geojsonList, file, indent=4)

    print(f"File saved as {filename}")

# Helper functions
def input_date_range(data):
    start_date = data['startDate']
    start_date = start_date.split("T")[0]
    end_date = data['endDate']
    end_date = end_date.split("T")[0]
    input_datetime_string = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    return input_datetime_string

def token_call():
    url = "https://services.sentinel-hub.com/oauth/token"
    headers = {"content-type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_secret": app.config['SENTINEL_CLIENT_SECRET'],
        "client_id":  app.config['SENTINEL_CLIENT_ID']
    }
    response = requests.post(url, headers=headers, data=data)  
    token = response.json()
   
    return token['access_token']

def cloud_cover_call(geojson_data, input_datetime_string,parameter):
    url = ""

    if parameter in ["Crop Growth Report", "Crop Stress (Biotic)"]:
        collection = "sentinel-2-l2a"
        url = "https://services.sentinel-hub.com/api/v1/catalog/1.0.0/search"
        
    else:
        collection = "landsat-ot-l1"
        url = "https://services-uswest2.sentinel-hub.com/api/v1/catalog/1.0.0/search" 
  
    # Convert geojson_data to a dictionary if it's a list
    if isinstance(geojson_data, list):
        # Wrap each item in the list as a feature in a FeatureCollection
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": geojson,
                    "properties": {}
                }
                for geojson in geojson_data
            ]
        }

    token = token_call()
    bearer = "Bearer " + token

    headers = {
        "Content-Type": "application/json",   
        "Authorization": bearer 
    } 
    data = {  
        "collections":[collection],
        "datetime": input_datetime_string, 
        "intersects": geojson_data['features'][0]['geometry'],   
        "limit": 100
    }  
    response = requests.post(url, headers=headers, json=data)

    response_content = response.json()
    print("response_content", response_content)
    
    date_dict = {}
    for feature in response_content["features"]:
        date = feature["properties"]["datetime"][:10]
        cloud_cover = feature["properties"]["eo:cloud_cover"]
        date_dict[date] = cloud_cover

    return date_dict


def compare(val, input_datetime_string, geojson_data,parameter):
    date_dict = cloud_cover_call(geojson_data, input_datetime_string,parameter)

    return {date: cloud_cover for date, cloud_cover in date_dict.items() if cloud_cover <= val}

@date_fetch_bp.route('/fetch_dates', methods=['POST'])
@jwt_required()
def fetch_dates():
    current_user_id = get_jwt_identity()
    data = request.get_json()

    if not data:
        return jsonify({"msg": "No data provided"}), 400
    
    try:
        startDate = data.get('startDate')
        endDate = data.get('endDate')
        parameter = data.get('selectedParameter')
        geojsonData = data.get('GeojsonData')
        print('geojsonData',geojsonData)
        Polygon_IDS = data.get('selectedFarmIds')

        if not all([startDate, endDate, parameter, geojsonData, Polygon_IDS]):
            return jsonify({"msg": "Missing required data fields"}), 400

        geojsonData = filter_geojson_by_polygon_ids(geojsonData, Polygon_IDS)
        print("after",geojsonData)
        centroid = get_centroid_of_geojson(geojsonData)
        input_datetime_string = input_date_range({"startDate": startDate, "endDate": endDate})
        available_dates = compare(100, input_datetime_string, geojsonData, parameter)
       
        return jsonify({"available_dates": available_dates, "msg": "Dates generated", "selectedFarms": geojsonData, "centroid" :centroid}), 200

    except (requests.exceptions.RequestException, Exception) as e:
        print("error:", str(e))
        return jsonify({"msg": "An error occurred, please try again", "error": str(e)}), 500


