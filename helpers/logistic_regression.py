from typing import List, Tuple
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


async def forward_propagate(inputs: List[float], weights: List[float], bias: float) -> float:
    linear_output = sum(i * w for i, w in zip(inputs, weights)) + bias
    return sigmoid(linear_output)


async def binary_cross_entropy(y_true: List[float], y_pred: List[float]) -> float:
    epsilon = 1e-15  # To prevent log(0)
    return -sum(y * math.log(y_pred + epsilon) + (1 - y) * math.log(1 - y_pred + epsilon) for y, y_pred in
                zip(y_true, y_pred)) / len(y_true)


async def train_logistic_regression(x_train: List[List[float]], y_train: List[float], learning_rate: float,
                                    epochs: int) -> Tuple[List[float], float]:
    weights = [0.0 for _ in range(len(x_train[0]))]
    bias = 0.0

    for epoch in range(epochs):
        for inputs, y_true in zip(x_train, y_train):
            y_pred = await forward_propagate(inputs, weights, bias)
            error = y_pred - y_true
            weights = [w - learning_rate * error * x for w, x in zip(weights, inputs)]
            bias -= learning_rate * error

        predictions = [await forward_propagate(x, weights, bias) for x in x_train]
        loss = await binary_cross_entropy(y_train, predictions)
        print(f"Epoch {epoch}, Loss: {loss}")

    return weights, bias
