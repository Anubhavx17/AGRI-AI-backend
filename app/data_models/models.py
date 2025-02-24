from datetime import datetime
from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.dialects.postgresql import JSON



#                               TRUNCATE TABLE public.crop_stress_graph_model CASCADE; ALTER SEQUENCE crop_stress_graph_model_id_seq RESTART WITH 1;
#                                    TRUNCATE TABLE public.result_table CASCADE;ALTER SEQUENCE result_table_id_seq RESTART WITH 1;

#                              select * from public.crop_stress_graph_model;


#                             SELECT id, user_id, tiff, excel, tiff_min_max, redsi_min_max, selected_date, selected_parameter,legend_quantile,project_id FROM public.result_table;


class UserModel(db.Model):
    __tablename__ = 'user_table'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    confirmed = db.Column(db.Boolean, default=False)
    projects = db.relationship('ProjectDetailsModel', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class ProjectDetailsModel(db.Model):
    __tablename__ = 'project_details_table'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_table.id'), nullable=False)
    project_title = db.Column(db.String(255), nullable=False)
    project_type = db.Column(db.String(100), nullable=False)
    starting_date = db.Column(db.String, nullable=False)
    selected_crops = db.Column(db.Text, nullable=False)
    farm_area = db.Column(db.String, nullable=True)
    number_of_farms = db.Column(db.Text, nullable=False)
    report_type = db.Column(db.String(100), nullable=False)
    geojson = db.Column(JSON, nullable=False)
    selected_parameters = db.Column(db.Text, nullable=False)
    selected_location = db.Column(db.String(255), nullable=True)  

class GraphModel(db.Model):
    __tablename__ = 'graph_table'
    id = db.Column(db.Integer, primary_key=True)
    unique_farm_id = db.Column(db.String, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user_table.id'), nullable=False)
    geojson = db.Column(db.JSON, nullable=False)
    selected_date = db.Column(db.String, nullable=False)
    selected_parameter = db.Column(db.Text, nullable=False)
    result_details = db.Column(db.String, nullable=False)

    # Foreign key linking to ResultTable
    result_id = db.Column(db.Integer, db.ForeignKey('result_table.id'), nullable=True)

class ResultModel(db.Model):
    __tablename__ = 'result_table'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user_table.id'), nullable=False)
    tiff = db.Column(db.String, nullable=False)
    excel = db.Column(db.String, nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project_details_table.id'), nullable=False)
    tiff_min_max = db.Column(db.String, nullable=True)
    selected_date = db.Column(db.String, nullable=False)
    selected_parameter = db.Column(db.Text, nullable=False)
    geojson = db.Column(db.JSON, nullable=False)

    graph = db.relationship('GraphModel', backref='result_table', lazy=True)
