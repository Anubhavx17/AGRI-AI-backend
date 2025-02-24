from app.data_models.models import UserModel, GraphModel,ResultModel
from minio import Minio
from minio.error import S3Error
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
from app import db
from app.main.helpers.helpers import get_presigned_url,upload_file_to_bucket,BUCKET_NAME


def result_already_exists(selected_date, selected_parameter, project_id, user_id):
    try:
        existing_result = ResultModel.query.filter(
            ResultModel.selected_date == selected_date,
            ResultModel.selected_parameter == selected_parameter,
            ResultModel.project_id == project_id,
            ResultModel.user_id == user_id
        ).first()
    
        if existing_result:
            print("Result already exists for the provided parameters.")
            response_data = {
            "id": existing_result.id,
            "tiff_url": get_presigned_url(BUCKET_NAME, f"{existing_result.id}.tiff"),
            "excel_url": get_presigned_url(BUCKET_NAME, f"{existing_result.id}.xlsx"),
            "tiff_min_max": existing_result.tiff_min_max,
            "redsi_min_max": existing_result.redsi_min_max,
            "geojson": existing_result.geojson,
            "legend_quantile":existing_result.legend_quantile
            }
            print("Response data prepared and sent.")
            return jsonify(response_data), 200
        else:
            return None  # Explicitly return None if no result is found

    except Exception as e:
        print(f"Error while checking existing results: {e}")
        return jsonify({"error": "Internal server error while checking existing results"}), 500

def create_result_entry(user_id, tiff_min_max, selected_date, selected_parameter, geojson,project_id,
                        tiff_path,excel_path):
    
    
    # Convert the numpy array to a string
    tiff_min_max_str = np.array2string(tiff_min_max, separator=',')

    
    # Step 1: Create an entry in the ResultModel
    new_result = ResultModel(user_id=user_id, tiff='', excel='', tiff_min_max=tiff_min_max_str,selected_date=selected_date, selected_parameter=selected_parameter,
                              geojson =geojson,project_id=project_id)  # Store as a string
    db.session.add(new_result)
    db.session.commit()  # Commit to get the auto-incremented ID

    # Step 2: Use the ID to construct filenames
    result_id = new_result.id
    tiff_filename = f"{result_id}.tiff"
    excel_filename = f"{result_id}.xlsx"

    # Save the graph result with its respective result id into graph db
    # save_temp_df_to_db(temp_df,result_id)

    # Step 3: Upload TIFF file
    with open(tiff_path, "rb") as tiff_file:
        tiff_url = upload_file_to_bucket(tiff_file, tiff_filename)
        
    # Step 4: Upload Excel file
    with open(excel_path, "rb") as excel_file:
        excel_url = upload_file_to_bucket(excel_file, excel_filename)

    # Step 5: Update the ResultModel entry with the URLs
    new_result.tiff = tiff_url
    new_result.excel = excel_url

    db.session.commit()
    print("result table entry created")
    return result_id
