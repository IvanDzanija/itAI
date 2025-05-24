from math import log2
import sys


def entropy(data):
    totalCount = 0
    for value in data.values():
        totalCount += value

    if totalCount == 0:
        return 0

    entropy = 0.0
    for value, count in data.items():
        probability = count / totalCount
        entropy -= probability * log2(probability)

    return entropy


def informationGain(data, feature, target):
    totalCount = len(data[target])
    if totalCount == 0:
        return 0

    featureValues = data[feature]
    featureCounts = dict()
    for value in featureValues:
        if value not in featureCounts:
            featureCounts[value] = 0
        featureCounts[value] += 1

    totalEntropy = entropy(data[target])
    weightedEntropy = 0.0

    for value, count in featureCounts.items():
        subset = {k: v for k, v in data.items() if v[feature] == value}
        subsetEntropy = entropy(subset[target])
        weightedEntropy += (count / totalCount) * subsetEntropy

    return totalEntropy - weightedEntropy


class Node:
    def __init__(self, feature=None, isLeaf=False):
        self.isLeaf = isLeaf
        self.feature = feature



class ID3:
    def __init__(self, target, features, depth=float("inf")):
        self.target = target
        self.features = features
        self.maxDepth = depth
        self.resultingTree = None

        self.entropyCache = dict()
        self.featureSplit = dict()
        self.featuresCategories = dict()
        self.expectedEntropy = dict()

        self.target = None
        self.features = None

    def categorizeFeatures(self, data):
        if self.features is None:
            self.features = list(data.keys())

        for feature in self.features:
            if feature not in self.featuresCategories:
                self.featuresCategories[feature] = dict()
            if feature not in self.entropyCache:
                self.entropyCache[feature] = dict()

            for value in data[feature]:
                if value not in self.featuresCategories[feature]:
                    self.featuresCategories[feature][value] = dict()
                if value not in self.entropyCache[feature]:
                    self.entropyCache[feature][value] = 0

            for row, value in enumerate(data[self.target]):
                if value not in self.featuresCategories[feature][data[feature][row]]:
                    self.featuresCategories[feature][data[feature][row]][value] = 0
                self.featuresCategories[feature][data[feature][row]][value] += 1

            for value in data[feature]:
                self.entropyCache[feature][value] = entropy(
                    self.featuresCategories[feature][value]
                )

        # print(self.entropyCache)

        return 0

    def entropyFeatures(self, data):
        totalCount = len(data[self.target])
        if totalCount == 0:
            return 0

        if self.features is None:
            # Should not happen, but just for linting
            self.features = list(data.keys())

        for feature in self.features:
            if feature not in self.featureSplit:
                self.featureSplit[feature] = {}
            if feature not in self.expectedEntropy:
                self.expectedEntropy[feature] = 0

            currentValues = data[feature]
            for value in currentValues:
                print(f"Processing feature: {feature}, value: {value}")
                if value not in self.featureSplit[feature]:
                    self.featureSplit[feature][value] = 0
                self.featureSplit[feature][value] += 1

            self.expectedEntropy[feature] = sum(
                self.entropyCache[feature][value] * count / totalCount
                for value, count in self.featureSplit[feature].items()
            )

        print(self.expectedEntropy)
        return 0

    def mostCommonValue(self, data, feature):
        if feature not in self.featuresCategories:
            return None

        maxCount = -1
        mostCommon = str()
        for value, count in self.featuresCategories[feature].items():
            if count > maxCount:
                maxCount = count
                mostCommon = value
            if count == maxCount:
                mostCommon = max(mostCommon,value)

        return mostCommon

    def fit(self, data, parent, features=None, link=None, depth=0):
        if depth == 0:
            if self.categorizeFeatures(data):
                print("Error categorizing features!")
            if self.entropyFeatures(data):
                print("Error calculating entropy!")

        if len(data) == 0:



        return 0

    def predict(self, data):
        return 0


class Parser:
    def __init__(self):
        self.path = None

    def setPath(self, path):
        self.path = path

    def parseToDataframe(self, dataframe):
        if self.path is None:
            print("Path not set. Use setPath()!")
            return None

        if not isinstance(dataframe, dict):
            print("Dataframe must be a dictionary!")
            return None

        with open(self.path, "r") as file:
            lines = file.readlines()
        features = lines[0].strip().split(",")
        target = features[-1]

        for feature in features:
            dataframe[feature] = []

        for line in lines[1:]:
            values = line.strip().split(",")
            for i, feature in enumerate(features):
                dataframe[feature].append(values[i])

        return target


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
    trainingDF = dict()
    target = parser.parseToDataframe(trainingDF)
    if trainingDF is None:
        print("Error parsing training set.")
        return 1

    # Parse the test set
    parser.setPath(test_set_path)
    testDF = dict()
    test_target = parser.parseToDataframe(testDF)
    if testDF is None:
        print("Error parsing test set.")
        return 1

    # Initialize the ID3 algorithm
    id3 = ID3(depth=tree_depth, target=target, features=list(trainingDF.keys()))
    id3.fit(trainingDF, target)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
