from threading import Thread
from flask import Flask, jsonify, request
from rio_viz.app import Client  # Import the Client class
from typing import Dict
import time
# Initialize the Flask app for the client server
app = Flask(__name__)

# Data structures
clients = {}  # Map of selected_date to Client instances
base_port = 8081  # Starting port number
initialized_ports = []  # Keep track of used ports

def get_next_port():
    """
    Dynamically allocate the next available port.
    """
    if initialized_ports:
        return max(initialized_ports) + 1  # Increment from the highest used port
    return base_port  # Start with the base port

def initialize_clients(tiff_data):
    """
    Initialize clients for given TIFF data and start servers on dynamically allocated ports.
    """
    start_time = time.time()  # Start the timer
    
    for tiff in tiff_data:
        selected_date = tiff["selected_date"]
        tiff_path = tiff["tiff_url"]
        tiff_min_max = tiff["tiff_min_max"]

        # if selected_date in clients:
        #     continue  # Skip if already initialized

        port = get_next_port()  # Get the next available port
        client = Client(src_path=tiff_path, port=port, host="127.0.0.1", config={})
        clients[selected_date] = {"client": client, "tiff_min_max": tiff_min_max}
        initialized_ports.append(port)

        thread = Thread(target=client.start)
        thread.daemon = True
        thread.start()
        print(f"Initialized client for {selected_date} on port {port}")
    
    end_time = time.time()  # End the timer
    print(f"Time taken to initialize clients: {end_time - start_time:.2f} seconds")



@app.route("/initialize", methods=["POST"])
def initialize():
    """
    Initialize clients for given TIFF data. Data sent via POST.
    """
    start_time = time.time()  # Start the timer

    tiff_data = request.get_json()
    initialize_clients(tiff_data)

    end_time = time.time()  # End the timer
    print(f"Time taken for /initialize endpoint: {end_time - start_time:.2f} seconds")

    return jsonify({"message": "Clients initialized successfully.", "ports": initialized_ports})

@app.route("/get_client_port", methods=["POST"])
def get_client_port():
    """
    Return the port for the given selected_date.
    """
    data = request.get_json()
    selected_date = data.get("selected_date")
    
    if not selected_date:
        selected_date,client_data = list(clients.items())[-1] # "2024-11-02" and  # {"client": {"port": 5002}, "tiff_min_max": [10, 200]}
        client_port = client_data["client"].port  # Access the port for the selected client
        tiff_min_max = client_data["tiff_min_max"]
        return jsonify({"selected_date": selected_date, "port": client_port, "tiff_min_max": tiff_min_max}), 200
    # Check if the date exists in the initialized clients
    if selected_date in clients:
        client_data = clients[selected_date]
        client_port = client_data["client"].port  # Access the port for the selected client
        tiff_min_max = client_data["tiff_min_max"]
        return jsonify({"selected_date": selected_date, "port": client_port, "tiff_min_max": tiff_min_max}), 200
    else:
        return jsonify({"error": f"No client found for date: {selected_date}"}), 404

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
