import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(yTrue, yPred):
    return np.mean((yTrue - yPred) ** 2)


# Initialize random generator
rng = np.random.default_rng()


class NeuralNetwork:
    def __init__(self, archEncoded, inputSize):
        self.weights = []
        self.biases = []
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
        for i in range(len(architecture) - 1):
            self.weights.append(
                rng.normal(0, 0.01, (architecture[i], architecture[i + 1]))
            )
            self.biases.append(rng.normal(0, 0.01, (1, architecture[i + 1])))

    def __str__(self):
        ret = ""
        for i in range(len(self.weights)):
            layer = {
                "weights": self.weights[i],
                "biases": self.biases[i],
            }
            ret += f"Layer {i + 1}:\n"
            ret += f"Weights:\n {layer['weights']}\n"
            ret += f"Biases:\n {layer['biases']}\n"

        return ret

    def forward(self, inputs):
        outputs = inputs
        for i in range(len(self.weights) - 1):
            currentWeights = self.weights[i]
            currentBiases = self.biases[i]
            outputs = sigmoid(outputs @ currentWeights + currentBiases)

        outputs = outputs @ self.weights[-1] + self.biases[-1]
        return outputs
