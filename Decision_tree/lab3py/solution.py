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

    labels = Counter([row[target] for row in data])
    print(labels)
    maxValue = max(labels.values())
    return min(key for key, value in labels.items() if value == maxValue)


class Node:
    def __init__(
        self,
        isLeaf=False,
        label=None,
        feature=None,
        mostCommonChild=None,
        children=None,
        depth=0,
    ):
        self.feature = feature
        self.children = children
        self.isLeaf = isLeaf
        if not isLeaf:
            self.children = dict()
        self.label = label
        self.mostCommonChild = mostCommonChild
        self.depth = depth

    def __str__(self):
        if self.isLeaf:
            print(str(self.label))
        else:
            print(str(self.feature))

    def setSubtrees(self, children, value):
        if isinstance(self.children, dict):
            self.children[value] = children


class ID3:
    def __init__(self, target, features, depth=float("inf")):
        self.target = target
        self.allFeatures = features
        self.maxDepth = depth
        self.resultingTree = Node()

    def fit(self, data):
        self.resultingTree = self.buildTree(data, data, set(self.allFeatures), 0)
        return 0

    def findBestFeature(self, data, features):
        maxValue = 0
        infos = defaultdict(float)
        for feature in features:
            infos[feature] = informationGain(data, feature, self.target)

        maxValue = max(infos.values())
        return min(key for key, value in infos.items() if value == maxValue)

    def filterData(self, data, feature, value):
        ret = list()
        for row in data:
            if row[feature] == value:
                ret.append(row)

        return ret

    def traverseTree(self, node, link):
        if node.isLeaf:
            print(link + node.label)
            return
        else:
            for value, child in node.children.items():
                add = str(node.depth) + ":" + str(node.feature) + "=" + str(value) + " "
                self.traverseTree(child, link + add)

    def printBranches(self):
        print("[BRANCHES]:")
        self.traverseTree(self.resultingTree, "")

    def buildTree(self, data, parentData, features, depth=0):
        print(f"[BUILDING_TREE] Depth: {depth}, Features: {features}")
        print(f"Data :{data}")
        if not data:
            return Node(isLeaf=True, label=mostCommon(parentData, self.target))

        labels = set([row[self.target] for row in data])
        print("Labels:", labels)
        if len(labels) == 1:
            return Node(isLeaf=True, label=labels.pop())

        v = mostCommon(data, self.target)
        print("Most common label:", v)
        if not features or depth == self.maxDepth:
            return Node(isLeaf=True, label=v)

        nextFeature = self.findBestFeature(data, features)
        print("Best feature:", nextFeature)
        retNode = Node(
            feature=nextFeature, isLeaf=False, depth=depth + 1, mostCommonChild=v
        )

        values = set(row[nextFeature] for row in data)
        nextFeatures = features - {nextFeature}
        for value in sorted(values):
            print(f"Processing value: {value} for feature: {nextFeature}")
            newData = self.filterData(data, nextFeature, value)
            retNode.setSubtrees(
                self.buildTree(newData, data, nextFeatures, depth + 1), value
            )

        return retNode

    def predictSingle(self, data):
        node = self.resultingTree
        while not node.isLeaf:
            value = data[node.feature]
            if node.children is None:
                return node.label
            if value in node.children:
                node = node.children[value]
            else:
                return node.mostCommonChild
        return node.label

    def printAccuracy(self, predictions, correct):
        if len(predictions) != len(correct):
            print("Not all predictions and calculated!")

        ans = 0
        for index, prediction in enumerate(predictions):
            if correct[index] == prediction:
                ans += 1
        print("[ACCURACY]:", end=" ")
        print(f"{ans / len(predictions):.5f}")

    def printConfusionMatrix(self, predictions, correct):
        print("[CONFUSION_MATRIX]:")
        matrix = defaultdict(lambda: defaultdict(int))
        keys = set()
        for index, prediction in enumerate(predictions):
            current = correct[index]
            keys.add(current)
            keys.add(prediction)
            matrix[current][prediction] += 1

        for key1 in sorted(keys):
            for key2 in sorted(keys):
                print(matrix[key1][key2], end=" ")
            print()

    def predict(self, data):
        correct = list()
        predictions = list()
        print("[PREDICTIONS]:", end=" ")
        for row in data:
            prediction = self.predictSingle(data=row)
            predictions.append(prediction)
            correct.append(row[self.target])
        print()
        self.printAccuracy(predictions, correct)
        self.printConfusionMatrix(predictions, correct)

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

    assert target == test_target

    # Initialize the ID3 algorithm
    modelID3 = ID3(
        target=target, features=set(trainingDF[0].keys()) - {target}, depth=tree_depth
    )
    modelID3.fit(trainingDF)
    modelID3.printBranches()
    modelID3.predict(testDF)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
