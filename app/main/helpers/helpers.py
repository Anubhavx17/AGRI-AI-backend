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

