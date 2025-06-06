import neural_network as nn
import numpy as np


class GeneticAlgorithm:
    def __init__(self, nnArch, popSize, elitism, mutProbability, inputs, K, iterations):
        self.nnArch = nnArch
        self.inputs = inputs
        self.popSize = popSize
        self.elitism = elitism
        self.mutationProbability = mutProbability
        self.K = K
        self.iterations = iterations
        self.initializePopulation()

    def initializePopulation(self):
        self.population = [
            nn.NeuralNetwork(self.nnArch, self.inputs) for _ in range(self.popSize)
        ]

    def evaluateFitness(self, trainData, trainTargets):
        self.errors = [
            nn.mse(net.forward(trainData), trainTargets) for net in self.population
        ]
        self.fScores = [1 / max(error, 1e-10) for error in self.errors]

    def getElites(self):
        selected = np.argsort(self.errors)[: self.elitism]
        return [self.population[i] for i in selected]

    def mutate(self, gene):
        rng = np.random.default_rng()
        for layer in gene.layers:
            rows = len(layer["weights"])
            for i in range(rows):
                cols = len(layer["weights"][i])
                for j in range(cols):
                    # Mutate weights with probability p
                    if rng.uniform(0, 1) < self.mutationProbability:
                        layer["weights"][i][j] += rng.normal(0, self.K)

            cols = len(layer["biases"])
            for j in range(cols):
                # Mutate biases with probability p
                if rng.uniform(0, 1) < self.mutationProbability:
                    layer["biases"][0][j] += rng.normal(0, self.K)
        return gene

    def crossover(self, parent1, parent2):
        child = nn.NeuralNetwork(self.nnArch, self.inputs)
        layersCount = len(parent1.layers)
        for i in range(layersCount):
            weights1 = parent1.layers[i]["weights"]
            weights2 = parent2.layers[i]["weights"]
            child.layers[i]["weights"] = (weights1 + weights2) / 2

            biases1 = parent1.layers[i]["biases"]
            biases2 = parent2.layers[i]["biases"]
            child.layers[i]["biases"] = (biases1 + biases2) / 2

        return child

    def chooseParents(self, numberOfParents=2):
        rng = np.random.default_rng()
        total = np.sum(self.fScores)
        probabilities = [score / total for score in self.fScores]
        selected = rng.choice(
            len(
                self.population
            ),  # Peaks into np.arrange and returns half-open interval [0, len(self.population))
            size=numberOfParents,
            p=probabilities,
        )
        # print(selected)
        return [self.population[i] for i in selected]

    def simulateDarwin(self, trainData, trainTarget):
        for iter in range(1, self.iterations + 1):
            self.evaluateFitness(trainData, trainTarget)

            if iter % 2000 == 0:
                print(f"[Train error @{iter}]: {min(self.errors)}")

            newPopulation = self.getElites()
            while len(newPopulation) < self.popSize:
                parent1, parent2 = self.chooseParents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                newPopulation.append(child)

            self.population = newPopulation

    def train(self, trainData, trainTarget):
        self.simulateDarwin(trainData, trainTarget)

    def test(self, testData, testTarget):
        self.evaluateFitness(testData, testTarget)
        print(f"[Test error]: {min(self.errors)}")
