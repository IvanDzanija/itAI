import neural_network as nn
import numpy as np
import argparse
from parser import Parser


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--train", required=True)
    argparser.add_argument("--test", required=True)
    argparser.add_argument("--nn", required=True)
    argparser.add_argument("--popsize", type=int, required=True)
    argparser.add_argument("--elitism", type=int, required=True)
    argparser.add_argument("--p", type=float, required=True)
    argparser.add_argument("--K", type=float, required=True)
    argparser.add_argument("--iter", type=int, required=True)

    args = argparser.parse_args()
    trainPath = args.train
    testPath = args.test
    nnArchitecture = args.nn
    popSize = args.popsize
    elitism = args.elitism
    mutationProbability = args.p
    K = args.K
    iterations = args.iter

    # Load training and testing data
    parser = Parser()

    parser.setPath(trainPath)
    trainDF = list()
    trainTarget = parser.parseToDataframe(trainDF)
    if trainTarget is None:
        print("Error: Could not parse training data.")
        return 1

    parser.setPath(testPath)
    testDF = list()
    testTarget = parser.parseToDataframe(testDF)
    if testTarget is None:
        print("Error: Could not parse testing data.")
        return 1

    assert testTarget == trainTarget

    return 0


if __name__ == "__main__":
    main()
