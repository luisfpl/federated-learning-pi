import flwr as fl

# Inicia el servidor federado
if __name__ == "__main__":
    fl.server.start_server(server_address="localhost:8080", config={"num_rounds": 3})
