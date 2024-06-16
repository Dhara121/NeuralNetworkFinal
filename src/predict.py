import numpy as np
from src.config import config
from src.preprocessing.data_management import load_model

def predict(X):
    model = load_model("two_input_xor_nn.pkl")
    theta0 = model["params"]["biases"]
    theta = model["params"]["weights"]
    f = model["activations"]

    z = [None] * config.NUM_LAYERS
    h = [None] * config.NUM_LAYERS
    h[0] = X

    def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
        return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs, current_layer_neurons_weights)

    def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
        if current_layer_neurons_activation_function == "linear":
            return current_layer_neurons_weighted_sums
        elif current_layer_neurons_activation_function == "sigmoid":
            return 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
        elif current_layer_neurons_activation_function == "tanh":
            return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums)) / \
                   (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))
        elif current_layer_neurons_activation_function == "relu":
            return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)

    for l in range(1, config.NUM_LAYERS):
        z[l] = layer_neurons_weighted_sum(h[l - 1], theta0[l], theta[l])
        h[l] = layer_neurons_output(z[l], f[l])

    return h[config.NUM_LAYERS - 1]

def binarize_predictions(predictions, threshold=0.5):
    return (predictions > threshold).astype(int)

if __name__ == "__main__":
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = predict(X_test)
    binary_predictions = binarize_predictions(predictions)
    print("Predictions for XOR function (binary):")
    for i, prediction in enumerate(binary_predictions):
        print(f"Input: {X_test[i]}, Prediction: {prediction}")

