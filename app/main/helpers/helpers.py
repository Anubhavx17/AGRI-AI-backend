from minio import Minio
from datetime import datetime, timedelta
import io
import os
from minio.error import S3Error
BUCKET_NAME = "dcmbucket" 

eos_client = Minio(
    'objectstore.e2enetworks.net',
    access_key='Q3VYHB4PV6GUW7CAGELM',  # Your access key
    secret_key='2G52C4LTQ5CYKCJXLMJALCLGI1CUNNLDLRAHPENC',  # Your secret key
    secure=True  # Use HTTPS
)

def get_presigned_url(bucket_name, object_name, expiration=timedelta(hours=1)):
    try:
        # Generate a presigned URL valid for 1 hour (default)
        presigned_url = eos_client.presigned_get_object(bucket_name, object_name, expires=expiration)
        return presigned_url
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
    
def upload_file_to_bucket(file_obj, file_name):
    try:
        # Determine the file size based on the type of file_obj
        if isinstance(file_obj, io.BytesIO):
            file_size = file_obj.getbuffer().nbytes
        else:
            file_size = os.fstat(file_obj.fileno()).st_size
        
        # Upload the file to the bucket
        eos_client.put_object(
            bucket_name=BUCKET_NAME,
            object_name=file_name,
            data=file_obj,
            length=file_size,
            content_type='application/octet-stream'
        )
        
        # Construct the URL for the uploaded file
        file_url = f"https://objectstore.e2enetworks.net/{BUCKET_NAME}/{file_name}"
        print("files uploaded")
        return file_url
    except S3Error as e:
        print(f"Error occurred while uploading: {e}")
        return None

FILES_TO_CREATE = [
    'ET.tiff', 'excel.xlsx', 'DateList.json', 'geojson_data.json', 
    'PCI_seasonal_stats.json', 'PCI_single_stats.json', 'Percentage.json', 
    'RoadQuality.json', 'RoadQualityData.json', 'Roads.json', 
    'Seasonal_PCI_Excel.xlsx', 'Single_PCI_Excel.xlsx', 'Surface_Stats.json', 
    'WrongDate.json'
]


def initialize_user_folder(user_id):
    folder_name = f"{user_id}/"
    # Check if the folder already exists
    result = eos_client.list_objects(bucket_name=BUCKET_NAME, prefix=folder_name)
    if 'Contents' in result:
        print(f"Folder {folder_name} already exists.")
    else:
        # Create folder and initialize files
        for filename in FILES_TO_CREATE:
            name = f"{user_id}/{filename}"
            eos_client.put_object(bucket_name=BUCKET_NAME, object_name=name, data='')  # Initialize empty file

# def upload_file_to_s3(file_content, user_id, filename, content_type='application/octet-stream'):
#     key = f"{user_id}/{filename}"
#     s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=file_content, ContentType=content_type)
#     return f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"

# def download_file_from_s3(user_id, filename):
#     key = f"{user_id}/{filename}"
#     # print(key)
#     try:
#         # print(f"Downloading from key: {key}")  # Debug print to check the key
#         obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
#         return BytesIO(obj['Body'].read())
#     except s3.exceptions.NoSuchKey:
#         print(f"Error: The specified key does not exist: {key}")
#         return None
#     except (NoCredentialsError, PartialCredentialsError) as e:
#         print(f"Credentials error: {str(e)}")
#         return None











import os
import json
import psycopg2
import base64
import pandas as pd

# Database connection settings
DB_CONFIG = {
    "user": "postgres",
    "password": "a4iawsrds",
    "host": "my-postgresdb-instance.cyungdgugllm.us-east-1.rds.amazonaws.com",
    "port": "5432",
    "database": "initial_db"
}

# Function to convert Excel file to base64
def encode_excel_to_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

# Function to update records in DB
def update_records(folder_path):
    selected_date = os.path.basename(folder_path)  # Extract date from folder name (e.g., "2025-02-23")
    
    geojson_path = os.path.join(folder_path, "GeojsonData.json")
    excel_path = os.path.join(folder_path, "water_stress.xlsx")

    # Check if files exist
    if not os.path.exists(geojson_path) or not os.path.exists(excel_path):
        print("Missing files in folder:", folder_path)
        return

    # Read GeoJSON
    with open(geojson_path, "r") as geojson_file:
        geojson_data = json.load(geojson_file)

    # Encode Excel file to Base64
    excel_base64 = encode_excel_to_base64(excel_path)

    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Update query: Matches dates like "2025-02-23T00:00:00.000Z"
        update_query = '''
            UPDATE "kzkAI"
            SET geojson = %s, excel = %s
            WHERE "selectedDate"::text LIKE %s;
        '''
        cursor.execute(update_query, (json.dumps(geojson_data), excel_base64, selected_date + '%'))

        # Commit and close
        connection.commit()
        print(f"Updated records for {selected_date}")

    except Exception as e:
        print("Database error:", e)

    finally:
        cursor.close()
        connection.close()

# Example Usage:
folder_to_process = "C:/path_to_folder/2025-02-23"  # Change this to the correct folder path
update_records(folder_to_process)
