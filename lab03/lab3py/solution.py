from math import log2
from collections import Counter, defaultdict
from parser import Parser
import sys


def entropy(data, target):
    if not data:
        return 0

    total = len(data)
    labels = [row[target] for row in data]
    counts = Counter(labels)
    return -sum((count / total) * log2(count / total) for count in counts.values())


def informationGain(data, feature, target):
    totalEntropy = entropy(data, target)
    features = defaultdict(list)

    for row in data:
        features[row[feature]].append(row)

    return totalEntropy - sum(
        len(feature) / len(data) * entropy(feature, target)
        for feature in features.values()
    )


def mostCommon(data, target):
    if not data:
        return None

    labels = [row[target] for row in data]
    return Counter(labels).most_common(1)[0][0]


class Node:
    def __init__(self, isLeaf, label=None, feature=None, children=None):
        self.feature = feature
        self.children = children
        self.isLeaf = isLeaf
        self.label = label


class ID3:
    def __init__(self, target, features, depth=float("inf")):
        self.target = target
        self.allFeatures = features
        self.maxDepth = depth
        self.resultingTree = None

    def fit(self, data):
        self.resultingTree = self.buildTree(data, data, set(self.allFeatures), 0)
        return 0

    def buildTree(self, data, parent_data, features, depth):
        if not data:
            return Node(isLeaf=True, label=mostCommon(parent_data, self.target))

    def predict(self, data):
        return 0


def main(args):
    argc = len(args)
    if argc < 2:
        print("Usage: python solution.py <path to training set> <path to test set>")
        return 1

    training_set_path = args[0]
    test_set_path = args[1]
    tree_depth = -1

    if argc > 2:
        tree_depth = int(args[2])

    # Initialize the parser
    parser = Parser()

    # Parse the training set
    parser.setPath(training_set_path)
    trainingDF = list()
    target = parser.parseToDataframe(trainingDF)
    if trainingDF is None:
        print("Error parsing training set.")
        return 1

    # Parse the test set
    parser.setPath(test_set_path)
    testDF = list()
    test_target = parser.parseToDataframe(testDF)
    if testDF is None:
        print("Error parsing test set.")
        return 1

    # Initialize the ID3 algorithm
    modelID3 = ID3(
        target=target, features=set(trainingDF[0].keys()) - {target}, depth=tree_depth
    )
    modelID3.fit(trainingDF)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
