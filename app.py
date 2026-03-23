from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import ast
import os

app = Flask(__name__)


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SingleNeuron(X, W):
    weightSum = W[0]
    for i in range(len(X)):
        weightSum += X[i] * W[i+1]
    return Sigmoid(weightSum)

def SingleLayerNetwork(X, W):
    return [SingleNeuron(X, neuron_weights) for neuron_weights in W]

def PredictDigit(X, W):
    outputs = SingleLayerNetwork(X, W)
    predicted = int(np.argmax(outputs))
    return outputs, predicted

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

def ReadTrainingSets(path):
    result = {}       # { "0": [[...],[...]], "1": [...], ... }
    current_digit = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.isdigit():
                current_digit = line
                if current_digit not in result:
                    result[current_digit] = []
            elif line.startswith("[") and line.endswith("]") and current_digit is not None:
                nums = [int(x) for x in line[1:-1].split(",")]
                result[current_digit].append(nums)
    return result

# ── Load once on startup ──
WEIGHTS_PATH = "TrainedWeights.txt"
TRAINING_PATH = "Training.txt"

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"{WEIGHTS_PATH} not found. Run ANN.py first.")

W = LoadWeights(WEIGHTS_PATH)
INPUT_SIZE = len(W[0]) - 1
print("Expected input size:", INPUT_SIZE)

# ── Routes ──

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data is None or "pixels" not in data:
        return jsonify({"error": "Missing 'pixels' field"}), 400

    pixels = data["pixels"]
    if len(pixels) != INPUT_SIZE:
        return jsonify({"error": f"Expected {INPUT_SIZE} pixels, got {len(pixels)}"}), 400

    X = [float(v) for v in pixels]
    outputs, predicted = PredictDigit(X, W)
    return jsonify({"outputs": outputs, "predicted_digit": predicted})

@app.route("/training-data", methods=["GET"])
def get_training_data():
    if not os.path.exists(TRAINING_PATH):
        return jsonify({}), 200
    data = ReadTrainingSets(TRAINING_PATH)
    return jsonify(data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)