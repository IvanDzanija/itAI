import neural_network as nn
import numpy as np


rng = np.random.default_rng()


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
        layers = len(gene.weights)
        for i in range(layers):
            rows = len(gene.weights[i])
            for j in range(rows):
                cols = len(gene.weights[i][j])
                for k in range(cols):
                    # Mutate weights with probability p
                    if rng.uniform(0, 1) < self.mutationProbability:
                        gene.weights[i][j][k] += rng.normal(0, self.K)

            cols = len(gene.biases[i][0])
            for j in range(cols):
                # Mutate biases with probability p
                if rng.uniform(0, 1) < self.mutationProbability:
                    gene.biases[i][0][j] += rng.normal(0, self.K)

    def crossover(self, parent1, parent2):
        child = nn.NeuralNetwork(self.nnArch, self.inputs)

        layers = len(child.weights)
        for i in range(layers):
            rows = len(child.weights[i])
            for j in range(rows):
                cols = len(child.weights[i][j])
                for k in range(cols):
                    child.weights[i][j][k] = (
                        parent1.weights[i][j][k] + parent2.weights[i][j][k]
                    ) / 2

            cols = len(child.biases[i][0])
            for j in range(cols):
                # Mutate biases with probability p
                child.biases[i][0][j] = (
                    parent1.biases[i][0][j] + parent2.biases[i][0][j]
                ) / 2

        return child

    def chooseParents(self, numberOfParents=2):
        selected = rng.choice(
            len(
                self.population
            ),  # Peaks into np.arrange and returns half-open interval [0, len(self.population))
            size=numberOfParents,
            p=self.fScores / np.sum(self.fScores),
        )
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
                self.mutate(child)
                newPopulation.append(child)

            self.population = newPopulation

    def train(self, trainData, trainTarget):
        self.simulateDarwin(trainData, trainTarget)

    def test(self, testData, testTarget):
        self.evaluateFitness(testData, testTarget)
        print(f"[Test error]: {min(self.errors)}")
