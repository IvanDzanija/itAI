import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(yTrue, yPred):
    return np.mean((yTrue - yPred) ** 2)


class NeuralNetwork:
    def __init__(self, archEncoded, inputSize):
        self.archDecode(archEncoded, inputSize)
        self.init_layers(self.architecture)

    def archDecode(self, archEncoded, inputSize):
        if archEncoded == "5s":
            self.architecture = [inputSize, 5, 1]
        elif archEncoded == "20s":
            self.architecture = [inputSize, 20, 1]
        elif archEncoded == "5s5s":
            self.architecture = [inputSize, 5, 5, 1]
        else:
            self.architecture = None
            raise ValueError(
                f"Unknown architecture encoding: {archEncoded}. "
                "Valid encodings are '5s', '20s', and '5s5s'."
            )

    def init_layers(self, architecture):
        self.layers = [dict() for _ in range(len(architecture) - 1)]

        # Initialize random generator
        rng = np.random.default_rng()

        for index, size in enumerate(architecture):
            if index >= len(self.layers):
                break
            currentLayer = dict()

            # Initialize weights and biases for the current layer
            currentLayer["weights"] = rng.normal(
                0, 0.01, (size, architecture[index + 1])
            )
            currentLayer["biases"] = rng.normal(0, 0.01, (1, architecture[index + 1]))
            self.layers[index] = currentLayer

    def __repr__(self):
        return f"NeuralNetwork(layers={self.layers})"

    def __str__(self):
        ret = "NeuralNetwork with layers:\n"
        for i, layer in enumerate(self.layers):
            ret += f"Layer {i + 1}:\n"
            ret += f"Weights:\n {layer['weights']}\n"
            ret += f"Biases:\n {layer['biases']}\n"

        return ret

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers[:-1]:
            currentWeights = layer["weights"]
            currentBiases = layer["biases"]
            outputs = sigmoid(np.dot(outputs, currentWeights) + currentBiases)

        outputs = (
            np.dot(outputs, self.layers[-1]["weights"]) + self.layers[-1]["biases"]
        )
        return outputs
