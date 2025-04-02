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

    for client in clients:
        tiff_url = client["tiff_url"]
        port = client['port_id']
        client = Client(src_path=tiff_url, port=port, host="127.0.0.1", config={})

        thread = Thread(target=client.start)
        thread.daemon = True
        thread.start()

        print(f"Initialized client on port {port}")

    end_time = time.time()  # End the timer
    print(f"Time taken for /initialize endpoint: {end_time - start_time:.2f} seconds")

    return jsonify({"message": "Clients initialized successfully."})

@app.route("/shutdown_clients", methods=["POST"])
def shutdown():
    start_time = time.time()  # Start the timer
    clients = request.get_json()

    for client in clients:
        tiff_url = client["tiff_url"]
        port = client['port_id']
        client = Client(src_path=tiff_url, port=port, host="127.0.0.1", config={})
        client.shutdown()  # calls something like ServerManager.shutdown_server("127.0.0.1:port")
        print(f"Shut down client on port {port}")

    end_time = time.time()  # End the timer
    print(f"Time taken for /shutdown endpoint: {end_time - start_time:.2f} seconds")

    return jsonify({"message": "Clients shut successfully."})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)