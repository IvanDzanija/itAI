# lab4py/solution.py

import argparse
import numpy as np
import csv

# ---------------------------- Neural Network ----------------------------


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class NeuralNetwork:
    def __init__(self, input_dim, architecture):
        self.input_dim = input_dim
        self.architecture = architecture
        self.layers = self.build_architecture(architecture)
        self.weight_shapes = self.get_weight_shapes()

    def build_architecture(self, arch):
        if arch == "5s":
            return [5]
        elif arch == "20s":
            return [20]
        elif arch == "5s5s":
            return [5, 5]
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    def get_weight_shapes(self):
        shapes = []
        layer_dims = [self.input_dim] + self.layers + [1]
        for i in range(len(layer_dims) - 1):
            shapes.append((layer_dims[i], layer_dims[i + 1]))  # weights
            shapes.append((1, layer_dims[i + 1]))  # biases
        return shapes

    def get_num_weights(self):
        return sum(np.prod(shape) for shape in self.weight_shapes)

    def set_weights(self, weights):
        self.params = []
        idx = 0
        for shape in self.weight_shapes:
            size = np.prod(shape)
            param = weights[idx : idx + size].reshape(shape)
            self.params.append(param)
            idx += size

    def forward(self, X):
        out = X
        for i in range(0, len(self.params) - 2, 2):
            W, b = self.params[i], self.params[i + 1]
            out = sigmoid(out @ W + b)
        W_out, b_out = self.params[-2], self.params[-1]
        return out @ W_out + b_out

    def predict(self, X):
        return self.forward(X)


# ---------------------------- Genetic Algorithm ----------------------------


def initialize_population(pop_size, num_weights):
    return np.random.normal(0, 0.01, (pop_size, num_weights))


def fitness(nn, X, y, pop):
    errors = np.array(
        [mean_squared_error(y, nn.set_weights(chrom) or nn.predict(X)) for chrom in pop]
    )
    return 1 / (errors + 1e-8), errors


def select(pop, fitnesses):
    probs = fitnesses / np.sum(fitnesses)
    indices = np.random.choice(len(pop), size=len(pop), p=probs)
    return pop[indices]


def crossover(parents):
    half = len(parents) // 2
    children = []
    for i in range(half):
        p1, p2 = parents[2 * i], parents[2 * i + 1]
        child = (p1 + p2) / 2
        children.append(child)
    return np.array(children)


def mutate(pop, p, K):
    noise = np.random.normal(0, K, pop.shape)
    mask = np.random.rand(*pop.shape) < p
    pop[mask] += noise[mask]
    return pop


def evolve(nn, X_train, y_train, args):
    num_weights = nn.get_num_weights()
    pop = initialize_population(args.popsize, num_weights)

    for gen in range(1, args.iter + 1):
        fit, errors = fitness(nn, X_train, y_train, pop)
        elite_idx = np.argsort(errors)[: args.elitism]
        elite = pop[elite_idx]

        selected = select(pop, fit)
        children = crossover(selected)
        mutated = mutate(children, args.p, args.K)

        pop = np.vstack((elite, mutated))

        if gen % 2000 == 0 or gen == args.iter:
            best_err = errors[elite_idx[0]]
            print(
                f"[{'Train' if gen < args.iter else 'Test'} error @{gen}]: {best_err:.6f}"
            )

    best_weights = pop[elite_idx[0]]
    return best_weights


# ---------------------------- Data Loading ----------------------------


def load_dataset(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        data = np.array([[float(x) for x in row] for row in reader])
    return data[:, :-1], data[:, -1:]


# ---------------------------- Main ----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--nn", required=True)
    parser.add_argument("--popsize", type=int, required=True)
    parser.add_argument("--elitism", type=int, required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--K", type=float, required=True)
    parser.add_argument("--iter", type=int, required=True)

    args = parser.parse_args()

    X_train, y_train = load_dataset(args.train)
    X_test, y_test = load_dataset(args.test)

    nn = NeuralNetwork(input_dim=X_train.shape[1], architecture=args.nn)

    best_weights = evolve(nn, X_train, y_train, args)

    nn.set_weights(best_weights)
    y_pred = nn.predict(X_test)
    test_error = mean_squared_error(y_test, y_pred)
    print(f"[Test error]: {test_error:.6f}")


if __name__ == "__main__":
    main()
