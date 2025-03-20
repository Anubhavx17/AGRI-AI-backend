import os 
import shutil
import requests
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.data_models.schemas import CropStressGraphModelSchema
from app.main.helpers.result_table_helpers import result_already_exists
from app.main.helpers.helpers import get_presigned_url, BUCKET_NAME
from app.data_models.models import ResultModel
from app.main.water_stress.SCRIPTS.water_stress_DASH import main

water_stress_bp = Blueprint('water_stress_bp', __name__)


### FUNCTION TO CREATE DIFFERENT FOLDERS FOR DIFFERENT USERS

def create_user_folder(user_id, folder_name, base_dir):
    """Creates a user-specific folder and copies the contents from the template folder."""

    user_folder = os.path.join(base_dir, f"{user_id}/{folder_name}") ## the folder to be created in the base dir
    source_folder = os.path.join(base_dir, folder_name) ## to be copied from the folder

    try:
        os.makedirs(user_folder, exist_ok=True)  # Ensure the directory exists

        if os.path.exists(source_folder):
            for item in os.listdir(source_folder):
                src_path = os.path.join(source_folder, item)
                dest_path = os.path.join(user_folder, item)

                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dest_path)

            print(f"Folder created for user {user_id} at {user_folder}")
        else:
            print("Error: Source folder does not exist.")
    except Exception as e:
        print(f"Error creating folder: {e}")


### FUNCTION TO DELETE USER FOLDERS

def delete_user_folder(user_id, base_dir):
    """Deletes the user-specific folder."""
    user_folder = os.path.join(base_dir, f"{user_id}") ## the folder to be created in the base dir

    try:
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
            print(f"Folder deleted for user {user_id}")
        else:
            print(f"Folder for user {user_id} does not exist.")
    except Exception as e:
        print(f"Error deleting folder: {e}")


### FUNCTION TO EXECUTE WATER STRESS SCRIPT

@water_stress_bp.route('/water_stress_api', methods = ['POST'])
@jwt_required()
def run_model():
        data = request.get_json()
        # print(data)
        user_id = get_jwt_identity()
        ## is result already exist return from here
        if result_already_exists(data.get('date'), 'Water Stress', data.get('project_id'), user_id):
                return jsonify({
            "status": "result_already_exists"
        }), 200
        
        create_user_folder(user_id, folder_name = "DL_CLOUD_MASKING", base_dir = 'backend/app/main/water_stress')
        create_user_folder(user_id, folder_name = "output_data", base_dir = 'backend/app/main')
    
        # initialize user wise folders - for dl_cloud_masking and output_data folder(tiff,excel) r locally
        # Call the main function with waters and use those local folders only
        # after that once the tiff and excel are uploaded to cloud delete them
        # existing_result_response = result_already_exists(data.get('date'), selected_parameter, project_id, user_id)

        main(data,user_id) ## call main function
        delete_user_folder(user_id, base_dir = 'backend/app/main/water_stress')
        delete_user_folder(user_id, base_dir = 'backend/app/main')

        # delete user wise folder(dl cloud masking) and output_data folder 
        return jsonify({
            "status": "Success"
        }), 200


###

@water_stress_bp.route('/send_all_result_data', methods = ['POST'])
@jwt_required()
def initialize_clients():
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
        f"?nodata=-9999&colormap_name=hsv&rescale={result.tiff_min_max}"
    )
        clients.append({
            'selected_date': result.selected_date,
            'tiff_url': presigned_tiff_url,
            'tiff_min_max': result.tiff_min_max,
            'excel_url': presigned_excel_url,
            'selected_parameter': result.selected_parameter,
            "geojson": result.geojson,
            "port_id":result.port_id,
             "tile_url": tile_url
        })

    response = requests.post("http://127.0.0.1:5001/initialize_clients", json=clients, timeout=30)

    try:
        response = requests.post("http://127.0.0.1:5001/initialize_clients", json=clients, timeout=30)
        response.raise_for_status()
        print("Clients initialized successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error initializing clients: {e}")
        return jsonify({"error": "Failed to initialize clients"}), 500

    return jsonify(clients)