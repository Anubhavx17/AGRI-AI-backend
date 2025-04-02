import os 
import shutil
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.main.helpers.result_table_helpers import result_already_exists
from app.main.crop_growth.crop_growth_DASH import main

crop_growth_bp = Blueprint('crop_growth_bp', __name__)
### FUNCTION TO CREATE DIFFERENT FOLDERS FOR DIFFERENT USERS

def create_user_folder(user_id, folder_name, base_dir):
    """Creates a user-specific folder and copies the contents from the template folder."""

    user_folder = os.path.join(base_dir, f"{user_id}/{folder_name}") ## the folder to be created in the base dir
    # source_folder = os.path.join(base_dir, folder_name) ## to be copied from the folder

    try:
        os.makedirs(user_folder, exist_ok = True)  # Ensure the directory exists
        print(f"Folder created for user {user_id} at {user_folder}")
        
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

@crop_growth_bp.route('/crop_growth_api', methods = ['POST'])
@jwt_required()
def run_model():
        data = request.get_json()
        # print(data)
        user_id = get_jwt_identity()
        ## is result already exist return from here
        if result_already_exists(data.get('date'), 'Crop Growth', data.get('project_id'), user_id):
                return jsonify({
            "status": "result_already_exists"
        }), 200
        
        create_user_folder(user_id, folder_name = "output_data", base_dir = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main')
    
        # initialize user wise folders - for dl_cloud_masking and output_data folder(tiff,excel) r locally
        # Call the main function with crops and use those local folders only
        # after that once the tiff and excel are uploaded to cloud delete them
        # existing_result_response = result_already_exists(data.get('date'), selected_parameter, project_id, user_id)

        main(data,user_id) ## call main function
        delete_user_folder(user_id, base_dir = r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main')

        # delete user wise folder(dl cloud masking) and output_data folder 
        return jsonify({
            "status": "Success"
        }), 200