import random
from typing import Dict, List, Tuple


async def initialize_network(num_inputs: int, num_hidden_neurons: int, num_outputs: int) -> Tuple[
    Dict[str, List[float]], Dict[str, List[float]]]:
    weights = {
        'hidden': [random.uniform(-1, 1) for _ in range(num_inputs * num_hidden_neurons)],
        'output': [random.uniform(-1, 1) for _ in range(num_hidden_neurons * num_outputs)]
    }
    biases = {
        'hidden': [random.uniform(-1, 1) for _ in range(num_hidden_neurons)],
        'output': [random.uniform(-1, 1) for _ in range(num_outputs)]
    }
    return weights, biases


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return 1 if x > 0 else 0


async def forward_propagate(inputs: List[float], weights: Dict[str, List[float]],
                            biases: Dict[str, List[float]]) -> float:
    # Compute the input to the hidden layer
    hidden_layer_input = [i * w for i, w in zip(inputs, weights['hidden'])]

    # Add the bias to each neuron in the hidden layer
    hidden_layer_input = [hli + bias for hli, bias in zip(hidden_layer_input, biases['hidden'])]

    # Apply the ReLU activation function
    hidden_layer_output = [relu(hli) for hli in hidden_layer_input]

    # Compute the input to the output layer
    output_layer_input = sum([h * w for h, w in zip(hidden_layer_output, weights['output'])]) + biases['output'][0]

    # The output layer uses a linear activation function, so the output is just the input to this layer
    output = output_layer_input
    return output


async def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


async def update_weights(inputs: List[float], output: float, y_true: float,
                         weights: Dict[str, List[float]], biases: Dict[str, List[float]],
                         learning_rate: float) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    # Calculate gradients
    output_error = y_true - output

    # Derivative of the ReLU for each neuron in the hidden layer
    hidden_layer_derivatives = [relu_derivative(inp) for inp in inputs]

    # Error for each neuron in the hidden layer
    hidden_layer_errors = [output_error * weight for weight in weights['output']]

    # Update weights and biases for the output layer
    weights['output'] = [w + learning_rate * output_error * output for w in weights['output']]
    biases['output'][0] += learning_rate * output_error

    # Update weights and biases for the hidden layer
    for i in range(len(weights['hidden'])):
        weights['hidden'][i] += learning_rate * hidden_layer_errors[i // len(inputs)] * hidden_layer_derivatives[
            i % len(inputs)] * inputs[i % len(inputs)]
        biases['hidden'][i // len(inputs)] += learning_rate * hidden_layer_errors[i // len(inputs)] * \
            hidden_layer_derivatives[i % len(inputs)]

    return weights, biases


async def train_network(x_train: List[List[float]], y_train: List[float], epochs: int, learning_rate: float,
                        num_hidden_neurons: int) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    num_inputs = len(x_train[0])
    num_outputs = 1

    weights, biases = await initialize_network(num_inputs, num_hidden_neurons, num_outputs)

    for epoch in range(epochs):
        for inputs, target in zip(x_train, y_train):
            output = await forward_propagate(inputs, weights, biases)
            weights, biases = await update_weights(inputs, output, target, weights, biases, learning_rate)

        if epoch % 100 == 0:
            predictions = [await forward_propagate(x, weights, biases) for x in x_train]
            loss = await mean_squared_error(y_train, predictions)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights, biases
