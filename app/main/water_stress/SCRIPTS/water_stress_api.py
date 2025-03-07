from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.data_models.schemas import CropStressGraphModelSchema
from app.main.helpers.result_table_helpers import result_already_exists
from app.main.helpers.helpers import initialize_user_folder
from app.main.water_stress.SCRIPTS.water_stress_DASH import main

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
            "status": "Success"
        }), 200