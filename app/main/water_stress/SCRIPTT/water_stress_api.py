from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.main.water_stress.SCRIPTT.water_stress_DASH import main
import os
import shutil
from app.main.helpers.result_table_helpers import result_already_exists

water_stress_bp = Blueprint('water_stress_api', __name__)

def create_user_folder(user_id,folder_name,base_dir):
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

def delete_user_folder(user_id,folder_name,base_dir):
    """Deletes the user-specific folder."""
    user_folder = os.path.join(base_dir, f"{user_id}/{folder_name}") ## the folder to be created in the base dir

    try:
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
            print(f"Folder deleted for user {user_id}")
        else:
            print(f"Folder for user {user_id} does not exist.")
    except Exception as e:
        print(f"Error deleting folder: {e}")


@water_stress_bp.route('/water_stress_api', methods=['POST'])
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
        
        create_user_folder(user_id,folder_name="DL_CLOUD_MASKING",base_dir = r"C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\water_stress")
        create_user_folder(user_id,folder_name="output_data",base_dir = r"C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main")
    
        # initialize user wise folders - for dl_cloud_masking and output_data folder(tiff,excel) r locally
        # Call the main function with parameters and use those local folders only
        # after that once the tiff and excel are uploaded to cloud delete them
        # existing_result_response = result_already_exists(data.get('date'), selected_parameter, project_id, user_id)

        main(data,user_id) ## call main function
        delete_user_folder(user_id,folder_name="DL_CLOUD_MASKING",base_dir = r"C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\water_stress")
        delete_user_folder(user_id,folder_name="output_data",base_dir = r"C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main")

        # delete user wise folder(dl cloud masking) and output_data folder 
        return jsonify({
            "status": "Success"
        }), 200


    

