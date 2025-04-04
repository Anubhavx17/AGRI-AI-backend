from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.data_models.models import ResultModel, GraphModel
from app.data_models.schemas import GraphModelSchema
from app import db
from shapely.geometry import shape
import json
import pandas as pd
from sqlalchemy import text
from sqlalchemy import func
from app.main.helpers.helpers import get_presigned_url,upload_file_to_bucket,BUCKET_NAME
from app.main.crop_stress.crop_stress_DASH import dict_to_gdf

summary_page_bp = Blueprint('summary_page_bp', __name__)
        

@summary_page_bp.route('/get_dates_analysed', methods=['POST'])
@jwt_required()
def get_dates_analysed():
    data = request.get_json()
    user_id = get_jwt_identity()
    project_id = data.get('project_id')


     # 1. Just get the count (single integer)
    project_count = db.session.query(
        func.count(ResultModel.id)
    ).filter(
        ResultModel.user_id == user_id,
        ResultModel.project_id == project_id
    ).scalar()

    # 2. Return it as JSON

    return jsonify({
        "project_count": project_count
    })


def group_by_parameter(user_id, project_id):
    try:
        grouped_results = (
            db.session.query(
                ResultModel.selected_parameter,
                func.array_agg(ResultModel.id).label("result_ids")
            )
            .filter(
                ResultModel.user_id == user_id,
                ResultModel.project_id == project_id,
            )
            .group_by(ResultModel.selected_parameter)
            .all()
        )
        return grouped_results
    except Exception as e:
        current_app.logger.error(f"Error in group_by_parameter: {str(e)}")

def group_dfs (graph_model_dfs,result_ids_by_param):
    # Create a schema instance for multiple objects.
    graph_schema = GraphModelSchema(many=True)

    for param, result_ids in result_ids_by_param.items():
    # Query the GraphModel rows where result_id is in the given list.
        graph_models = GraphModel.query.filter(GraphModel.result_id.in_(result_ids)).all()
        
        # Serialize the graph models into a list of dictionaries.
        serialized_data = graph_schema.dump(graph_models)
        
        # Convert the serialized data into a Pandas DataFrame.
        df = pd.DataFrame(serialized_data)
        
        # Store the DataFrame in the dictionary with the parameter as key.
        graph_model_dfs[param] = df

    # Now, graph_model_dfs is a dictionary with keys 'Crop Stress', 'Crop Growth', 'Water Stress'
    # and the values are DataFrames containing the corresponding GraphModel data.
    return graph_model_dfs

@summary_page_bp.route('/get_best_worst_farms', methods=['POST'])
@jwt_required()
def get_best_worst_farms():
    data = request.get_json()
    user_id = get_jwt_identity()
    project_id = data.get('project_id')
    geojson_data = data.get('geojson_data')
    geojson_data_df = dict_to_gdf(geojson_data) # make it a gdf
   

     # 1. Group result IDs by selected_parameter
    grouped_results = group_by_parameter(user_id, project_id)
    if not grouped_results:
        return jsonify({"error": "No results found for provided user and project"}), 404

     # 2. Convert grouped_results (list of tuples) to a dictionary.
    result_ids_by_param = {param: ids for param, ids in grouped_results}


    # 3. Initialize a dictionary to hold the DataFrames for each parameter.
    graph_model_dfs = {}

    graph_model_dfs = group_dfs(graph_model_dfs,result_ids_by_param)
    
    # Extract lists for each parameter.
    cs_df = graph_model_dfs['Crop Stress']
    ws_df = graph_model_dfs['Water Stress']
    cg_df = graph_model_dfs['Crop Growth']
    
    # if not (cs_df or cg_df or ws_df):
    #         return jsonify({"error": "No Excel files found for any parameter"}), 404
    
    # 4. Process the excels using your best_worst function.
    best_worst_list = best_worst(cs_df, ws_df, cg_df,geojson_data_df)

    return jsonify({
        "best_worst_list": best_worst_list
    })


def best_worst(cs_df,ws_df,cg_df,geojson_data_df):
    print(cs_df.columns)
    df = geojson_data_df
    print(df,len(df))
    df['CS_COUNT1'] = " "
    df['CS_COUNT2'] = " "
    df['CS_CLOUD_COUNT'] = " "
    df['CS_SCORE'] = " "
    df['CG_COUNT1'] = " "
    df['CG_COUNT2'] = " "
    df['CG_COUNT3'] = " "
    df['CG_CLOUD_COUNT'] = " "
    df['CG_SCORE'] = " "
    df['WS_COUNT1'] = " "
    df['WS_COUNT2'] = " "
    df['WS_COUNT3'] = " "
    df['WS_CLOUD_COUNT'] = " "
    df['WS_SCORE'] = " "
    df['FARM_SCORE'] = " "

    def cs_score(str):
        flag = 1
        if str == 'Presence of Cloud' or str == 'None':
            flag = 0
            return 0, flag
        elif str == 'No Crop Stress':
            return 1, flag
        elif str == 'Severe Crop Stress':
            return 2, flag
        else:
        # Default case: return 0 and flag 1, or handle as needed
            return 0, flag
        
    def cg_score(str):
        flag = 1
        if str == 'Presence of Cloud' or str == 'None':
            flag = 0
            return 0, flag
        elif str == 'Ideal Crop Growth':
            return 1, flag
        elif str == 'Average Crop Growth':
            return 2, flag
        elif str == 'Poor Crop Growth':
            return 3, flag
        else:
        # Default case: return 0 and flag 1, or handle as needed
            return 0, flag
        
    def ws_score(str):
        flag = 1
        if str == 'Presence of Cloud' or str == 'None':
            flag = 0
            return 0, flag
        elif str == 'No Water Stress':
            return 1, flag
        elif str == 'Medium Water Stress':
            return 2, flag
        elif str == 'Severe Water Stress':
            return 3, flag
        else:
        # Default case: return 0 and flag 1, or handle as needed
            return 0, flag
        
    i = 0
    for farm_id, grouped_df_by_farmid in cs_df.groupby('unique_farm_id'):
        # rows = [row1, row2, row3, row4, row5, row6]
        if(farm_id == 'A0087S01314'):
            print(grouped_df_by_farmid)
        cs = 0
        cloud_cnt = 0
        count1 = 0
        count2 = 0
        for row in grouped_df_by_farmid.itertuples(index=False):
            
            val, flag = cs_score(row.inference)
            # print(val)
            if val == 1:
                count1 += 1
            elif val == 2:
                count2 += 1
            cs += val
            if flag == 0:
                cloud_cnt += 1
            
        # print(count2)
        # print(cs)
        df['CS_COUNT1'][i] = count1
        df['CS_COUNT2'][i] = count2
        df['CS_CLOUD_COUNT'][i] = cloud_cnt


        if(farm_id == 'A0087S01314'):
            print(count1, count2)
            print(df['CS_COUNT1'][i], df['CS_COUNT2'][i], df['CS_CLOUD_COUNT'][i])
        if cloud_cnt != len(grouped_df_by_farmid):
            cs = cs / (2 * (len(grouped_df_by_farmid) - cloud_cnt))
            df['CS_SCORE'][i] = round(cs, 3)
        else:
            df['CS_SCORE'][i] = 0
        # print(cs)
        i += 1
        if(farm_id == 'A0087S01314'):
            row1 = df[df['FARM_ID'] == 'A0087S01314']['CS_COUNT1']
            row2 = df[df['FARM_ID'] == 'A0087S01314']['CS_COUNT2']
            print(row1,row2)


    i = 0
    for farm_id, grouped_df_by_farmid in cg_df.groupby('unique_farm_id'):
        # rows = [row1, row2, row3, row4, row5, row6]
        cg = 0
        cloud_cnt = 0
        count1 = 0
        count2 = 0
        count3 = 0
        for row in grouped_df_by_farmid.itertuples(index=False):
            val, flag = cg_score(row.inference)
            if val == 1:
                count1 += 1
            elif val == 2:
                count2 += 1
            elif val == 3:
                count3 += 1
            cg += val
            if flag == 0:
                cloud_cnt += 1

        # print(cg)
        df['CG_COUNT1'][i] = count1
        df['CG_COUNT2'][i] = count2
        df['CG_COUNT3'][i] = count3
        df['CG_CLOUD_COUNT'][i] = cloud_cnt
        if cloud_cnt != len(grouped_df_by_farmid):
            cg = cg / (3 * (len(grouped_df_by_farmid) - cloud_cnt))
            df['CG_SCORE'][i] = round(cg, 3)
        else:
            df['CG_SCORE'][i] = 0
        # print(cg)
        i += 1

    i = 0
    for farm_id, grouped_df_by_farmid in ws_df.groupby('unique_farm_id'):
        # rows = [row1, row2, row3, row4, row5, row6]
        ws = 0
        cloud_cnt = 0
        count1 = 0
        count2 = 0
        count3 = 0
        for row in grouped_df_by_farmid.itertuples(index=False):
            val, flag = ws_score(row.inference)
            if val == 1:
                count1 += 1
            elif val == 2:
                count2 += 1
            elif val == 3:
                count3 += 1
            ws += val
            if flag == 0:
                cloud_cnt += 1

        # print(ws)
        df['WS_COUNT1'][i] = count1
        df['WS_COUNT2'][i] = count2
        df['WS_COUNT3'][i] = count3
        df['WS_CLOUD_COUNT'][i] = cloud_cnt
        if cloud_cnt != len(grouped_df_by_farmid):
            ws = ws / (3 * (len(grouped_df_by_farmid) - cloud_cnt))
            df['WS_SCORE'][i] = round(ws, 3)
        else:
            df['WS_SCORE'][i] = 0
        # print(ws)
        i += 1
    

    for i in range(len(df)):
        nodata_cnt = 0
        if df['CS_SCORE'][i] == 0:
            nodata_cnt += 1
        if df['CG_SCORE'][i] == 0:
            nodata_cnt += 1
        if df['WS_SCORE'][i] == 0:
            nodata_cnt += 1
            
        if nodata_cnt != 3:
            df['FARM_SCORE'][i] = round((df['CS_SCORE'][i] + df['CG_SCORE'][i] + df['WS_SCORE'][i]) / (3 - nodata_cnt), 3)
        else:
            df['FARM_SCORE'][i] = 0


    # Sort the DataFrame by the given scores (descending order)
    df = df.sort_values(by=['FARM_SCORE', 'CS_SCORE', 'WS_SCORE', 'CG_SCORE'],
                        ascending=[False, False, False, False])
    mean_score = df['FARM_SCORE'].sum() / len(df)

    print(mean_score)
    if mean_score < 0.34:
        print("Poor")
        mean_score = 'Poor'
    elif mean_score >= 0.34 and mean_score < 0.67:
        print("Average")
        mean_score = 'Average'
    else:
        print("Good") 
        mean_score = 'Good'

    # For best farms (bottom 5 rows after sorting), include FARM_ID
    best_subset = df.tail(5)[['FARM_ID', 'WS_COUNT1', 'CG_COUNT1', 'CS_COUNT1']]
    print("Best Farms (FARM_ID, WS_COUNT1, CG_COUNT1, CS_COUNT1):")
    print(best_subset)

    # For worst farms (top 5 rows after sorting), include FARM_ID
    worst_subset = df.head(5)[['FARM_ID', 'WS_COUNT3', 'CG_COUNT3', 'CS_COUNT2']]
    print("Worst Farms (FARM_ID, WS_COUNT3, CG_COUNT3, CS_COUNT2):")
    print(worst_subset)


    # Get bottom 5 rows (best farms with lower FARM_SCORE) and top 5 rows (worst farms)
    top5_df = df.tail(5)      # Best farms (lower scores)
    worst5_df = df.head(5)    # Worst farms (higher scores)

    # Extract FARM_ID lists from both DataFrames
    top5_ids = top5_df['FARM_ID'].tolist()    # Best farm IDs
    worst5_ids = worst5_df['FARM_ID'].tolist()  # Worst farm IDs

    print("Top 5 best farm IDs:", top5_ids)
    print("Top 5 worst farm IDs:", worst5_ids)

    # Initialize dictionaries for geojson results
    top5_geojson = {}
    worst5_geojson = {}

    # For each best farm_id, fetch its geojson from the GraphModel table.
    for farm_id in top5_ids:
        graph_row = GraphModel.query.filter_by(unique_farm_id=str(farm_id)).first()
        if graph_row:
            top5_geojson[farm_id] = graph_row.geojson

    # For each worst farm_id, fetch its geojson.
    for farm_id in worst5_ids:
        graph_row = GraphModel.query.filter_by(unique_farm_id=str(farm_id)).first()
        if graph_row:
            worst5_geojson[farm_id] = graph_row.geojson

    # Define the columns to retrieve for each category
    best_columns = ['WS_COUNT1', 'CG_COUNT1', 'CS_COUNT1']
    worst_columns = ['WS_COUNT3', 'CG_COUNT3', 'CS_COUNT2']

    # Create dictionaries to store the count values for each farm.
    # For best farms (using best_columns) and worst farms (using worst_columns)
    best_farm_values = {}
    worst_farm_values = {}

    # Process best farms (those in top5_ids, i.e. best farms)
    for _, row in df[df['FARM_ID'].isin(top5_ids)].iterrows():
        temp = {
            'ws': row['WS_COUNT1'],
            'cg': row['CG_COUNT1'],
            'cs': row['CS_COUNT1']
        }
        best_farm_values[row['FARM_ID']] = temp

    # Process worst farms (those in worst5_ids, i.e. worst farms)
    for _, row in df[df['FARM_ID'].isin(worst5_ids)].iterrows():
        temp = {
            'ws': row['WS_COUNT3'],
            'cg': row['CG_COUNT3'],
            'cs': row['CS_COUNT2']
        }
        worst_farm_values[row['FARM_ID']] = temp

    print("Best Farm Values:")
    print(best_farm_values)
    print("\nWorst Farm Values:")
    print(worst_farm_values)


    df.to_excel(r'C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\main\insights\FINAL5.xlsx', index = False)
    
    best_worst_list = []
    best_worst_list.append({
            "top_5": top5_geojson,
            "worst_5": worst5_geojson,
            "farm_health": mean_score,
            "best_farm_values":best_farm_values,
            "worst_farm_values":worst_farm_values
        })

    return best_worst_list

