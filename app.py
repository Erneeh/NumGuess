from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import ast
import os

app = Flask(__name__)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SingleNeuron(X, W):
    """
    X: list of pixel values (length N)
    W: list of weights (length N+1) -> [bias, w0, w1, ..., wN-1]
    """
    weightSum = W[0]  # bias
    for i in range(len(X)):
        weightSum += X[i] * W[i+1]
    return Sigmoid(weightSum)

def SingleLayerNetwork(X, W):
    """
    X: input vector
    W: list of neurons' weights (10 x (N+1))
    """
    return [SingleNeuron(X, neuron_weights) for neuron_weights in W]

def PredictDigit(X, W):
    outputs = SingleLayerNetwork(X, W)
    predicted = int(np.argmax(outputs))
    return outputs, predicted

# --------- LOAD TRAINED WEIGHTS ---------

def LoadWeights(path="TrainedWeights.txt"):
    W = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Output layer"):
                continue
            if line.startswith("[") and line.endswith("]"):
                neuron_weights = ast.literal_eval(line)
                W.append(neuron_weights)
    print(f"Loaded {len(W)} neurons from {path}")
    return W

# Load once when server starts
WEIGHTS_PATH = "TrainedWeights.txt"
if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"{WEIGHTS_PATH} not found. Train in ANN.py and call SaveWeights(W) first."
    )

W = LoadWeights(WEIGHTS_PATH)
INPUT_SIZE = len(W[0]) - 1  # N = weights - 1 bias
print("Expected input size:", INPUT_SIZE)

# --------- ROUTES ---------

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data is None or "pixels" not in data:
        return jsonify({"error": "Missing 'pixels' field in JSON"}), 400

    pixels = data["pixels"]

    if not isinstance(pixels, list):
        return jsonify({"error": "'pixels' must be a list"}), 400

    if len(pixels) != INPUT_SIZE:
        return jsonify({
            "error": f"Expected {INPUT_SIZE} pixels, got {len(pixels)}"
        }), 400

    # convert to float
    X = [float(v) for v in pixels]

    outputs, predicted = PredictDigit(X, W)

    return jsonify({
        "outputs": outputs,
        "predicted_digit": predicted
    })

if __name__ == "__main__":
    app.run(debug=True)


# PASIJUNGT ->
#pip install flask
#pip install numpy flask
#python app.py

