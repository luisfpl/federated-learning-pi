# Federated Learning on Raspberry Pi

![Python](https://img.shields.io/badge/Python-v3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0-red)
![Flower](https://img.shields.io/badge/Flower-1.3.0-green)
![RaspberryPi](https://img.shields.io/badge/RaspberryPi-Edge-00aaff)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Distributed-orange)
![MIT License](https://img.shields.io/badge/License-MIT-yellow)

## Overview

Federated Learning on Raspberry Pi is a project that demonstrates how to implement a federated learning model across multiple Raspberry Pi devices. The goal is to perform distributed training of a machine learning model without sharing the underlying data, thereby ensuring privacy and security.

## Features

- **Federated Learning**: Collaborative training across multiple devices without sharing raw data.
- **Edge Computing**: Leverage Raspberry Pi's capabilities for running machine learning models at the edge.
- **Modular Design**: Easily adaptable to different models and datasets.
- **Privacy-Preserving**: Data remains on the local device, only model parameters are shared.

## Requirements

- **Hardware**: Raspberry Pi (preferably Raspberry Pi 4)
- **Software**: Python 3.x, pip, internet connection
- **Libraries**: PyTorch, Flower, torchvision

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/FederatedLearningPi.git
   cd FederatedLearningPi
   ```

2. **Install the Required Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage

### Training the Model on Raspberry Pi

To start training the model on your Raspberry Pi, execute the following command:

```bash
python3 train.py
```

This script will:
- Load the MNIST dataset.
- Train a simple Convolutional Neural Network (CNN) on your local device.
- Communicate with the federated learning server to synchronize model parameters.

### Running the Federated Learning Server

Run the following command on your server or central coordinating machine:

```bash
python3 server.py
```

This script will:
- Initialize the federated learning server.
- Coordinate the training process across multiple Raspberry Pi devices.
- Aggregate and distribute model updates from the clients.

### Federated Learning Workflow

1. **Server Initialization**: The central server starts and waits for clients to connect.
2. **Client Training**: Each Raspberry Pi (client) trains the model locally with its own data.
3. **Model Aggregation**: The server aggregates updates from each client to improve the global model.
4. **Model Distribution**: The updated global model is sent back to clients for the next round of training.

## Project Structure

```bash
.
├── train.py               # Client-side training script
├── server.py              # Federated learning server script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Contribution

We welcome contributions! Here's how you can contribute:

1. Fork the project.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

![Back to top](#top)
