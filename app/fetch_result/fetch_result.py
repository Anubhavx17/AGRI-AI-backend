from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.data_models.models import ResultModel  # Import necessary modules and models
from sqlalchemy import desc
from minio import Minio
from minio.error import S3Error
from datetime import datetime, timedelta

# Create a Blueprint for the result fetching routes
result_fetch_bp = Blueprint('result_fetch', __name__)

eos_client = Minio(
    'objectstore.e2enetworks.net',
    access_key='Q3VYHB4PV6GUW7CAGELM',  # Your access key
    secret_key='2G52C4LTQ5CYKCJXLMJALCLGI1CUNNLDLRAHPENC',  # Your secret key
    secure=True  # Use HTTPS
)

BUCKET_NAME = "dcmbucket"

def get_presigned_url(bucket_name, object_name, expiration=timedelta(hours=1)):
    try:
        # Generate a presigned URL valid for 1 hour (default)
        presigned_url = eos_client.presigned_get_object(bucket_name, object_name, expires=expiration)
        return presigned_url
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None


@result_fetch_bp.route('/api/fetch_result', methods=['POST'])
@jwt_required()
def fetch_result():
    try:
        # Get the current user's ID
        data = request.get_json()
        user_id = get_jwt_identity()
        print("user_id", user_id)
        project_id = data.get('ProjectId')
        # Fetch the latest result for the current user
        latest_result = (
            ResultModel.query
            .filter_by(user_id=user_id, project_id=project_id)
            .order_by(ResultModel.id.desc())  # Order by id in descending order
            .first()  # Fetch the first (latest) result
        )
        if not latest_result:
            return jsonify({"message": "No results found for the user"}), 404

        tiff_presigned_url = get_presigned_url(BUCKET_NAME, f"{latest_result.id}.tiff")
        excel_presigned_url = get_presigned_url(BUCKET_NAME, f"{latest_result.id}.xlsx")
       
        # load_new_tiff(tiff_presigned_url,latest_result.tiff_min_max)
        # Build the response with the required fields
        result_data = {
            "id": latest_result.id,
            "tiff_url": tiff_presigned_url,
            "excel_url": excel_presigned_url,
            "tiff_min_max": latest_result.tiff_min_max,
            "redsi_min_max": latest_result.redsi_min_max,
            "geojson": latest_result.geojson,
            "legend_quantile":latest_result.legend_quantile
        }

        return jsonify({"message": "Result fetched successfully", "data": result_data}), 200

    except Exception as e:
        return jsonify({"message": "An error occurred while fetching the result", "error": str(e)}), 500



