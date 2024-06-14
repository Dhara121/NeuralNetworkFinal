import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_model

app = Flask(__name__)

def predict(X):
    # Load trained model
    loaded_model = load_model("two_input_xor_nn.pkl")

    # Assuming loaded_model contains necessary parameters like theta0 and theta

    # Example of preprocessing steps
    preprocessor = pp.PreprocessData()
    preprocessor.fit(X)
    X_processed, _ = preprocessor.transform(X)

    # Example of prediction with mini-batch processing
    num_samples = len(X_processed)
    predictions = []

    for i in range(0, num_samples, config.MINI_BATCH_SIZE):
        X_batch = X_processed[i:i + config.MINI_BATCH_SIZE]

        # Example of forward propagation using loaded_model parameters
        # Adjust based on your actual model structure and prediction method
        Z = np.dot(X_batch, loaded_model["params"]["weights"][1]) + loaded_model["params"]["biases"][1]
        A = 1 / (1 + np.exp(-Z))  # Example activation function (sigmoid)
        predictions.extend(A)

    return predictions

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json['inputs']
    data = np.array(data)
    predictions = predict(data)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
