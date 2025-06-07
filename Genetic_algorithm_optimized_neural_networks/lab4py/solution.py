import numpy as np
import argparse
import gen_algo as ga
from parser import Parser


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
    trainTarget = parser.parseToMatrix(trainDF)
    trainTargets = [row[-1] for row in trainDF]
    trainDF = [row[:-1] for row in trainDF]

    if trainTarget is None:
        print("Error: Could not parse training data.")
        return 1

    parser.setPath(testPath)
    testDF = list()
    testTarget = parser.parseToMatrix(testDF)
    testTargets = [row[-1] for row in testDF]
    testDF = [row[:-1] for row in testDF]

    if testTarget is None:
        print("Error: Could not parse testing data.")
        return 1

    assert testTarget == trainTarget

    trainDF = np.array(trainDF)
    trainTargets = np.array(trainTargets).reshape(-1, 1)
    testDF = np.array(testDF)
    testTargets = np.array(testTargets).reshape(-1, 1)

    darwin = ga.GeneticAlgorithm(
        nnArch=nnArchitecture,
        popSize=popSize,
        elitism=elitism,
        K=K,
        iterations=iterations,
        mutProbability=mutationProbability,
        inputs=len(trainDF[0]),
    )
    darwin.train(trainDF, trainTargets)
    darwin.test(testDF, testTargets)

    return 0


if __name__ == "__main__":
    main()
