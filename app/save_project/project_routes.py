from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.data_models.models import UserModel, ProjectDetailsModel,ResultModel
from app.data_models.schemas import ProjectDetailsSchema
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
import ast

project_bp = Blueprint('project_bp', __name__)


from shapely.geometry import shape, MultiPolygon

def get_centroid_of_geojson(geojson_data):
    polygons = []

    # Iterate over each feature in the GeoJSON data
    for feature in geojson_data['features']:
        try:
            geometry = shape(feature['geometry'])

            # Check if the geometry is valid and not empty
            if geometry.is_valid and not geometry.is_empty:
                polygons.append(geometry)
            else:
                print(f"Warning: Invalid or empty geometry found in feature: {feature}")

        except Exception as e:
            print(f"Error processing geometry for feature: {feature}. Error: {e}")

    # Check if we have valid polygons collected
    if not polygons:
        print("Error: No valid geometries found in the GeoJSON data.")
        return None

    # Combine all valid polygons into a MultiPolygon
    combined_polygon = MultiPolygon(polygons)

    # Check if the combined polygon is empty
    if combined_polygon.is_empty:
        print("Error: The combined MultiPolygon is empty.")
        return None

    # Calculate the centroid of the combined geometry
    centroid = combined_polygon.centroid

    # Check if the calculated centroid is empty
    if centroid.is_empty:
        print("Error: The calculated centroid is empty.")
        return None

    return {'latitude': centroid.y, 'longitude': centroid.x}


def convert_geojsonlist_to_geodf(geojsonList):
    """
    Converts a list of GeoJSON-like geometries into a GeoDataFrame, ensuring that
    each polygon in a MultiPolygon gets its own Polygon_ID and preserves the original properties.

    Parameters:
    - geojsonList: A list of GeoJSON-like geometries (dictionaries).

    Returns:
    - A GeoDataFrame with the geometries, original properties, and a Polygon_ID column.
    """
    geometries = []
    properties_list = []

    for geo in geojsonList:
        # Check if the geo is a FeatureCollection
        if geo.get('type') == 'FeatureCollection':
            for feature in geo['features']:
                geom = shape(feature['geometry'])
                properties = feature.get('properties', {})  # Keep track of properties
                if isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        geometries.append(poly)
                        properties_list.append(properties)  # Append properties for each polygon
                elif isinstance(geom, Polygon):
                    geometries.append(geom)
                    properties_list.append(properties)  # Append properties for single polygon
        # Check if the geo is a Feature
        elif geo.get('type') == 'Feature':
            geom = shape(geo['geometry'])
            properties = geo.get('properties', {})  # Keep track of properties
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    geometries.append(poly)
                    properties_list.append(properties)  # Append properties for each polygon
            elif isinstance(geom, Polygon):
                geometries.append(geom)
                properties_list.append(properties)  # Append properties for single polygon
        else:
            # Handle MultiPolygon or Polygon directly
            geom = shape(geo)
            if isinstance(geom, MultiPolygon):
                for poly in geom.geoms:
                    geometries.append(poly)
                    properties_list.append({})  # If no properties, append empty dict
            elif isinstance(geom, Polygon):
                geometries.append(geom)
                properties_list.append({})  # If no properties, append empty dict
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(properties_list, geometry=geometries)  # Attach properties
    
    # Add the Polygon_ID column, with unique IDs for each polygon
    gdf['Polygon_ID'] = range(1, len(gdf) + 1)
    
    return gdf

@project_bp.route('/projects', methods=['GET'])
@jwt_required()
def get_projects():
    current_user_id = get_jwt_identity()

    # Fetch projects for the current user
    projects = ProjectDetailsModel.query.filter_by(user_id=current_user_id).all()

    # Serialize the project details
    schema = ProjectDetailsSchema(many=True)
    project_data = schema.dump(projects)

    # Get the count of project IDs in the ResultModel
    project_counts = (
        db.session.query(ResultModel.project_id, db.func.count(ResultModel.project_id).label('count'))
        .filter(ResultModel.project_id.in_([project['id'] for project in project_data]))
        .group_by(ResultModel.project_id)
        .all()
    )
    # Convert query result to a dictionary for quick lookup
    project_counts_dict = {project_id: count for project_id, count in project_counts}

    # Add centroid and project count to each project
    for project in project_data:
        if 'geojson' in project:  # Assuming 'geojson' is the key where GeoJSON data is stored
            project['centroid'] = get_centroid_of_geojson(project['geojson'])

        # Add the count of results for this project
        project['result_count'] = project_counts_dict.get(project['id'], 0)

    return jsonify(project_data), 200



@project_bp.route('/projects', methods=['POST'])
@jwt_required()
def save_project():
    current_user_id = get_jwt_identity()
    data = request.get_json()
    print(data)
    save_project_boolean = data.get('save_project')

    existing_project_with_title = ProjectDetailsModel.query.filter_by(
        user_id=current_user_id,
        project_title=data.get('project_title')
    ).first()

    if existing_project_with_title:
        return jsonify({"msg": "Project Title already exists", "titleExists": True}), 201
    
    if save_project_boolean :
        geojson = data.get('geojson')
        shp = convert_geojsonlist_to_geodf(geojson)
        geojsonString = shp.to_json()
        geojsonList = ast.literal_eval(geojsonString)

        project = ProjectDetailsModel(
            user_id=current_user_id,
            project_title=data.get('project_title'),
            project_type=data.get('project_type'),
            starting_date=data.get('starting_date'),
            selected_crops=data.get('selected_crops'),
            farm_area= data.get('farm_area'),
            number_of_farms= data.get('number_of_farms'),
            report_type=data.get('report_type'),
            geojson=geojsonList,
            selected_parameters=",".join(data.get('selected_parameters')),
            selected_location=data.get('selected_location'),
        )

        db.session.add(project)
        db.session.commit()
        return jsonify({"msg": "Project saved successfully!", "projectSaved": True}), 201
 
    return jsonify({"msg": "Project title unique, move fwd!", "titleExists": False}), 201

@project_bp.route('/projects/delete_current_user', methods=['DELETE'])
@jwt_required()
def delete_all_user_projects():
    current_user_id = get_jwt_identity()
    
    # Query all projects associated with the current user
    projects = ProjectDetailsModel.query.filter_by(user_id=current_user_id).all()
    
    if not projects:
        return jsonify({"msg": "No projects found for this user.", "deleted": False}), 404
    
    # Delete all projects
    for project in projects:
        db.session.delete(project)
    
    db.session.commit()
    
    return jsonify({"msg": "All projects deleted successfully.", "deleted": True}), 200


@project_bp.route('/projects/delete_all', methods=['DELETE'])
@jwt_required()  # Optional: Only allow authenticated users to perform this action
def delete_all_projects():
    # Query all projects
    projects = ProjectDetailsModel.query.all()
    
    if not projects:
        return jsonify({"msg": "No projects found.", "deleted": False}), 404
    
    # Delete all projects
    for project in projects:
        db.session.delete(project)
    
    db.session.commit()
    
    return jsonify({"msg": "All projects deleted successfully.", "deleted": True}), 200



# @project_bp.route('/clear', methods=['DELETE'])
# def clear_database():
#         db.session.query(ProjectDetails).delete()
#         db.session.query(User).delete()
#         db.session.commit()
#         return jsonify({"msg": "All data has been cleared successfully!"}), 200
