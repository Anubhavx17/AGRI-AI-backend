from app.data_models.models import UserModel, GraphModel,ResultModel
from minio import Minio
from minio.error import S3Error
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request
import pandas as pd
import json
import numpy as np
from app import db

def clean_json_data(data):
    """
    Recursively clean the JSON data by replacing inf, -inf, and NaN with None.
    """
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, float) and (np.isinf(data) or np.isnan(data)):
        return None
    return data

def generate_geojson(row):
    """
    Generate a GeoJSON polygon feature for a row using Lat_1, Long_1, Lat_2, Long_2, Lat_3, Long_3, Lat_4, Long_4
    and include all other properties from the row.
    """
    # Create a polygon feature using the latitude and longitude columns
    coordinates = [
        [row['Long_1'], row['Lat_1']],
        [row['Long_2'], row['Lat_2']],
        [row['Long_3'], row['Lat_3']],
        [row['Long_4'], row['Lat_4']],
        [row['Long_1'], row['Lat_1']]  # Closing the polygon loop
    ]
    
    # Add all other columns as properties except geometry
    properties = row.to_dict()
    
    # Create the GeoJSON structure
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates]
        },
        "properties": properties
    }
    
    return json.dumps(geojson)

def generate_custom_dataframe(final_df, input_date, parameter):
    """
    This function creates a new dataframe from final_df with the columns:
    Unique_ID, Result (combining REDSI and INFERENCE), Parameter, Date, and geojson_data.
    
    The geojson_data is generated using Lat_1, Long_1, Lat_2, Long_2, Lat_3, Long_3, Lat_4 columns and all
    other properties from final_df are included in the properties section of the GeoJSON.
    
    Args:
    - final_df: DataFrame containing the final data including latitude, longitude, and other properties.
    - input_date: Date string or datetime representing the date for the Date column.
    - parameter: Parameter string to store in the Parameter column.
    
    Returns:
    - temp_df: New DataFrame with the required columns and generated geojson_data.
    """
    
    # Create a new dataframe with the required columns
    temp_df = pd.DataFrame()

    # Create the Unique_ID column (assuming FARM_ID is used as a unique identifier)
    temp_df['Unique_ID'] = final_df['FARM_ID'] ## became farm id now

    # Create the Result column by combining REDSI and INFERENCE
    if parameter == 'Crop Stress Biotic':
        temp_df['Result'] = final_df.apply(lambda row: f"REDSI: {row['REDSI']}, INFERENCE: {row['INFERENCE']}", axis=1)
    
    elif parameter == 'Water Stress':
        temp_df['Result'] = final_df.apply(lambda row: f" SWSI: {row['SWSI']} , INFERENCE: {row['INFERENCE']} , ET:{row['ET']}", axis=1)

    else :
        ## third parameter
        temp_df['Result'] = final_df.apply(lambda row: f"REDSI: {row['REDSI']}, INFERENCE: {row['INFERENCE']}", axis=1)

    # Create a Parameter column
    temp_df['Parameter'] = parameter

    # Create a Date column using the input_date variable from the function arguments
    temp_df['Date'] = input_date

    # Create the Geojson_data column by applying the generate_geojson function to each row in final_df
    temp_df['geojson_data'] = final_df.apply(generate_geojson, axis=1)

    return temp_df

def save_temp_df_to_db(temp_df,result_id,user_id):
    """
    Iterates through temp_df and saves each row to the database using save_results.
    
    Args:
    - temp_df: A DataFrame containing the results to be saved in the database.
    
    Returns:
    - Success message after all rows are saved.
    """
   
    for index, row in temp_df.iterrows():
        # Prepare data for saving
        data = {
            'unique_farm_id': row['Unique_ID'],
            'geojson': json.loads(row['geojson_data']),  # Convert geojson_data string to JSON
            'selected_date': row['Date'],
            'selected_parameter': row['Parameter'],
            'result_details': row['Result'],
        }

        # Save the row using the save_results function
        save_results(data, result_id,user_id)
    print('project saved')
    return "done"


def save_results(data,result_id,user_id):
    """
    Saves the result data into the GraphModel.

    Args:
    - data: A dictionary containing the fields required to save to the GraphModel.

    Returns:
    - A success message.
    """
    existing_entry = GraphModel.query.filter_by(
        unique_farm_id=data.get('unique_farm_id'),
        selected_date=data.get('selected_date'),
        selected_parameter=data.get('selected_parameter'),
        result_id = result_id
    ).first()

    if existing_entry:
        # If an entry exists, return a message indicating that it already exists
        print("Entry already exists, no new record created")
        return jsonify({"msg": "Entry already exists, no new record created", "projectSaved": False}), 201

   
    cleaned_result_details = clean_json_data(data.get('result_details'))
    cleaned_geojson = clean_json_data(data.get('geojson'))
    geojson_str = json.dumps(cleaned_geojson)

    # Convert cleaned_result_details to a JSON string
    result_details_str = json.dumps(cleaned_result_details)

    stress_result = GraphModel(
        unique_farm_id=data.get('unique_farm_id'),
        user_id=user_id,
        # geojson=data.get('geojson'),
        geojson=geojson_str,
        selected_date=data.get('selected_date'),
        selected_parameter=data.get('selected_parameter'),
        # result_details=data.get('result_details'),
        result_details=result_details_str,  
        result_id=result_id
    )

    db.session.add(stress_result)
    db.session.commit()
    return jsonify({"msg": "Project saved successfully!", "projectSaved": True}), 201
