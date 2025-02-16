from datetime import datetime
from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.postgresql import JSON



#                               TRUNCATE TABLE public.crop_stress_graph_model CASCADE; ALTER SEQUENCE crop_stress_graph_model_id_seq RESTART WITH 1;
#                                    TRUNCATE TABLE public.result_table CASCADE;ALTER SEQUENCE result_table_id_seq RESTART WITH 1;

#                              select * from public.crop_stress_graph_model;


#                             SELECT id, user_id, tiff, excel, tiff_min_max, redsi_min_max, selected_date, selected_parameter,legend_quantile FROM public.result_table;


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    confirmed = db.Column(db.Boolean, default=False)
    projects = db.relationship('ProjectDetails', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ProjectDetails(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    project_title = db.Column(db.String(255), nullable=False)  # Define max length
    project_type = db.Column(db.String(100), nullable=False)  # Define max length
    starting_date = db.Column(db.String, nullable=False)
    selected_crops = db.Column(db.Text, nullable=False)
    farm_area = db.Column(db.String, nullable=True)
    number_of_farms = db.Column(db.Text, nullable=False)
    report_type = db.Column(db.String(100), nullable=False)  # Define max length
    geojson = db.Column(JSON, nullable=False)
    selected_parameters = db.Column(db.Text, nullable=False)
    selected_location = db.Column(db.String(255), nullable=True)  

class CropStressGraphModel(db.Model):
    __tablename__ = 'crop_stress_graph_model'  # Explicit table name

    id = db.Column(db.Integer, primary_key=True)
    unique_farm_id = db.Column(db.String, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    geojson = db.Column(db.JSON, nullable=False)
    selected_date = db.Column(db.String, nullable=False)
    selected_parameter = db.Column(db.Text, nullable=False)
    result_details = db.Column(db.String, nullable=False)
    
    # Foreign key linking to ResultTable
    result_id = db.Column(db.Integer, db.ForeignKey('result_table.id'), unique=False, nullable=True) ## not unique as multple rows can have same result_id, nullable true see logic how

class ResultTable(db.Model):
    __tablename__ = 'result_table'  # Explicit table name

    id = db.Column(db.Integer, primary_key=True)  # Auto-incremented primary key
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign key linking to a user
    tiff = db.Column(db.String, nullable=False)  # Path or URL to the TIFF file associated with this result
    excel = db.Column(db.String, nullable=False)  # Path or URL to the Excel file associated with this result
    project_id = db.Column(db.Integer, db.ForeignKey('project_details.id'), nullable=False)
    tiff_min_max = db.Column(db.String, nullable=True)
    redsi_min_max = db.Column(db.String, nullable=True)
    selected_date = db.Column(db.String, nullable=False)
    selected_parameter = db.Column(db.Text, nullable=False)
    geojson = db.Column(db.JSON, nullable=False)
    legend_quantile = db.Column(db.Text, nullable=True)
   # Establish relationship to CropStressGraphModel (one-to-many)
    graph = db.relationship('CropStressGraphModel', backref='result_table', lazy=True)




