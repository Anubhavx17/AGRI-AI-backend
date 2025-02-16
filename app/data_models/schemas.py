from marshmallow import Schema, fields

class ProjectDetailsSchema(Schema):
    id = fields.Int(dump_only=True)
    user_id = fields.Int(dump_only=True)
    project_title = fields.Str(required=True)
    project_type = fields.Str(required=True)
    starting_date = fields.Str(required=True)
    selected_crops = fields.Str(required=True)
    report_type = fields.Str(required=True)
    geojson = fields.Raw(required=True)  # Adjusted to match the model
    selected_parameters = fields.Str(required=True)
    selected_location = fields.Str(allow_none=True)
    farm_area = fields.Str(allow_none=True)  # Added to match the model
    number_of_farms = fields.Str(required=True)  # Added to match the model

class CropStressGraphModelSchema(Schema):
    id = fields.Int(dump_only=True)
    user_id = fields.Int(dump_only=True)
    unique_farm_id = fields.Str(dump_only=True)
    geojson = fields.Raw(required=True)  #Adjusted to match the model
    selected_date = fields.Str(required=True)
    selected_parameter = fields.Str(required=True)
    result_details = fields.Str(required=True)