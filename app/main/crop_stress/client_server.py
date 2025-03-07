from threading import Thread
from flask import Flask, jsonify, request
from rio_viz.app import Client
from concurrent.futures import ThreadPoolExecutor
import time

app = Flask(__name__)

""" Initialize clients for given TIFF data and start servers on allocated ports using port id"""


@app.route("/initialize_clients", methods=["POST"])
def initialize():
    start_time = time.time()  # Start the timer
    clients = request.get_json()

    def start_client(client_data):
        tiff_url = client_data["tiff_url"]
        port = client_data['port_id']
        client = Client(src_path=tiff_url, port=port, host="127.0.0.1", config={})
        client.start()
        print(f"Initialized client on port {port}")

    # Run all clients in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(start_client, clients)

    end_time = time.time()  # End the timer
    print(f"Time taken for /initialize endpoint: {end_time - start_time:.2f} seconds")

    return jsonify({"message": "Clients initialized successfully."})


# @app.route("/initialize_clients", methods=["POST"])
# def initialize():
#     start_time = time.time()  # Start the timer
#     clients = request.get_json()
#     ## instead of loop need to initialize ALL THE TIFFS SIMULTANOESLY TO reduce latency + error handling

#     for client in clients:
#         tiff_url = client["tiff_url"]
#         port = client['port_id']
#         client = Client(src_path=tiff_url, port=port, host="127.0.0.1", config={})

#         thread = Thread(target=client.start)
#         thread.daemon = True
#         thread.start()

#         print(f"Initialized client on port {port}")
    

#     end_time = time.time()  # End the timer
#     print(f"Time taken for /initialize endpoint: {end_time - start_time:.2f} seconds")

#     return jsonify({"message": "Clients initialized successfully."})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)

# # Data structures
# clients = {}  # Map of selected_date to Client instances
# base_port = 8081  # Starting port number
# initialized_ports = []  # Keep track of used ports


# """Dynamically allocate the next available port"""
# def get_next_port():
#     if initialized_ports:
#         return max(initialized_ports) + 1  # Increment from the highest used port
#     return base_port  # Start with the base port


# """ Initialize clients for given TIFF data and start servers on dynamically allocated ports """
# @app.route("/initialize_clients", methods=["POST"])
# def initialize():
#     start_time = time.time()  # Start the timer
#     tiff_data = request.get_json()
    
#     for tiff in tiff_data:
#         # selected_date = tiff["selected_date"]
#         tiff_url = tiff["tiff_url"]
#         port = tiff['port_id']
#         # selected_parameter = tiff["selected_parameter"]
#         # user_id = tiff['user_id']
#         # result_id = tiff['result_id']

#         # key = (user_id, selected_date, selected_parameter) # Unique Key

#         # if key in clients:
#         #     continue  # Skip if already initialized

#         # port = get_next_port()  # Get the next available port

#         client = Client(src_path=tiff_url, port=port, host="127.0.0.1", config={})
#         # clients[key] = {
#         #     "client": client,
#         #     "tiff_min_max": tiff_min_max
#         # }

#         # # clients[selected_date] = {"client": client, "tiff_min_max": tiff_min_max}
#         # initialized_ports.append(port)

#         thread = Thread(target=client.start)
#         thread.daemon = True
#         thread.start()

#         print(f"Initialized client on port {port}")
    

#     end_time = time.time()  # End the timer
#     print(f"Time taken for /initialize endpoint: {end_time - start_time:.2f} seconds")

#     return jsonify({"message": "Clients initialized successfully."})


# """ Return the port for the given selected_date """
# @app.route("/get_client_port", methods=["POST"])
# def get_client_port():

#     data = request.get_json()
#     key = data.get("key")
    
#     if not key:
#         key,client_data = list(clients.items())[-1] # "2024-11-02" and  # {"client": {"port": 5002}, "tiff_min_max": [10, 200]}
#         client_port = client_data["client"].port  # Access the port for the selected client
#         tiff_min_max = client_data["tiff_min_max"]
#         return jsonify({"key": key, "port": client_port, "tiff_min_max": tiff_min_max}), 200
    
#     # Check if the date exists in the initialized clients
#     if key in clients:
#         client_data = clients[key]
#         client_port = client_data["client"].port  # Access the port for the selected client
#         tiff_min_max = client_data["tiff_min_max"]
#         return jsonify({"key": key, "port": client_port, "tiff_min_max": tiff_min_max}), 200
#     else:
#         return jsonify({"error": f"No client found for date: {key}"}), 404

