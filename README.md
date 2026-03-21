# Digit Recognizer 🧠

A handwritten digit recognizer built from scratch — no ML libraries.
Draw a digit on a 10×10 grid and a neural network predicts what it is.

Built to understand how neural networks actually work under the hood,
not just call `.fit()`.

**Stack:** Python · Flask · Vanilla JS · Tailwind CSS

## How it works

- A single-layer neural network with 10 neurons (one per digit)
- Backpropagation implemented manually in `ANN.py` using only NumPy
- Trained on pixel grid data stored in `Training.txt`
- Weights saved to `TrainedWeights.txt` and loaded at server start
- Frontend sends a 100-pixel array to the Flask API on predict

## Limitations (by design)

This was built to learn, not to compete with MNIST.
Single layer + small training set = rough accuracy, but the math works.

## Getting Started

1. Install dependencies
```bash
   pip install numpy flask
   pip install flask
```
2. Start the server
```bash
   python app.py
```

3. Open `http://localhost:5000`, draw a digit, hit Predict.
