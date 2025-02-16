import time

import httpx


from rio_viz.app import Client

# Create rio-viz Client (using server-thread to launch backgroud task)
client = Client(r"C:\Users\ANUBHAV\OneDrive\Desktop\AGRI_DCM\backend\app\q\tiffs\32_bit.tiff")

# Gives some time for the server to setup

time.sleep(1)

r = httpx.get(
    f"{client.endpoint}/tilejson.json",
    params = {
        "rescale": "1600,2000",  # from the info endpoint
        "colormap_name": "hsv",
    }
).json()

bounds = r["bounds"]
print(r)
