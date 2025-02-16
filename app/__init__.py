from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_mail import Mail
from flask_migrate import Migrate
from flask_cors import CORS  # Import CORS
from config import Config

db = SQLAlchemy()
jwt = JWTManager()
mail = Mail()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    CORS(app, supports_credentials=True)
    app.config.from_object(Config)

    db.init_app(app)
    jwt.init_app(app)
    mail.init_app(app)
    migrate.init_app(app, db)

    # Apply CORS


    from app.save_project.project_routes import project_bp
    from app.login_auth.auth import auth_bp
    from app.date_fetch.date_fetch_routes import date_fetch_bp
    from app.main.crop_stress.crop_stress_routes import crop_stress_bp
    # from app.tile_server.tile import tile_server_bp
    # from app.main.fetch_result_table import fetch_result_bp
    from app.fetch_result.fetch_result import result_fetch_bp
    from app.main.water_stress.SCRIPTT.water_stress_api import water_stress_bp

    app.register_blueprint(date_fetch_bp)
    app.register_blueprint(crop_stress_bp)
    app.register_blueprint(project_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(water_stress_bp)
    # app.register_blueprint(tile_server_bp)
    app.register_blueprint(result_fetch_bp)
    return app
