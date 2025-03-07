import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=10)

    # SQLAlchemy Engine Options
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 7200  # Recycle connections after 1 hour
    }

    # SMTP Configuration for Amazon SES
    MAIL_SERVER = os.getenv('MAIL_SERVER')
    MAIL_PORT = int(os.getenv('MAIL_PORT'))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() in ['true', '1', 't']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')

    # Sentinel Hub Credentials
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    LANDSAT8_ID = os.getenv('LANDSAT8_ID')
    SENTINEL2_ID = os.getenv('SENTINEL2_ID')
    SENTINEL3_ID = os.getenv('SENTINEL3_ID')
    OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')