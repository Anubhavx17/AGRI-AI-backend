# from flask import Flask, jsonify, redirect, request, Blueprint
# from threading import Thread
# import subprocess
# import os
# import signal
# import time
# import httpx
# import rasterio
# import numpy as np
# import time
# from rio_viz.app import Client

# app = Flask(__name__)

# tile_server_bp = Blueprint('tile_server_bp', __name__)

# def load_new_tiff(tiff_path):
#     global rioclient, min_max_str

#     # Calculate min-max values
#     with rasterio.open(tiff_path) as src:
#         tiff = src.read()
#         tiff_min = np.amin(tiff)
#         tiff_max = np.amax(tiff)
#         min_max = [round(tiff_min, 2), round(tiff_max, 2)]

#     min_max_str = "[" + ", ".join(f"{v:.2f}".rstrip('0').rstrip('.') for v in min_max) + "]"
#     print(f"New TIFF loaded with min-max: {min_max_str}")

#     # Restart the Rio-Viz client with the new TIFF
#     if rioclient:
#         rioclient.shutdown()
    
#     rioclient = Client(tiff_path)
#     time.sleep(1)
#     return {"message": "TIFF loaded successfully"}

# @tile_server_bp.route('/fetchTile', methods=['GET'])
# def tilejson():
    
#     global rioclient, min_max_str
#     if not rioclient:
#         return jsonify({"error": "TIFF not loaded"}), 404

#     try:
#         # Fetch the tile JSON from the Rio-Viz client
#         r = httpx.get(
#             f"{rioclient.endpoint}/tilejson.json",
#             params={
#                 "rescale": min_max_str,
#                 "colormap_name": "viridis",
#                 "nodata": "0"
#             }
#         ).json()

#         # Return both the tile JSON and min-max values
#         return jsonify({"tiles": r, "min_max": min_max_str})
#     except Exception as e:
#         print(f"Error fetching tile JSON: {e}")
#         return jsonify({"error": "Failed to fetch tile JSON"}), 500
    