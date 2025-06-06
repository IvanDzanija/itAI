class Parser:
    def __init__(self):
        self.path = None

    def setPath(self, path):
        self.path = path

    def parseToMatrix(self, matrix):
        if self.path is None:
            print("Path not set. Use setPath()!")
            return None

        with open(self.path, "r") as file:
            lines = file.readlines()
        features = lines[0].strip().split(",")
        target = features[-1]

        for line in lines[1:]:
            values = line.strip().split(",")
            matrix.append([float(value) for value in values])

        return target

    def parseToDataframe(self, dataframe):
        if self.path is None:
            print("Path not set. Use setPath()!")
            return None

        if not isinstance(dataframe, list):
            print("Dataframe must be a list of dictionaries!")
            return None

        with open(self.path, "r") as file:
            lines = file.readlines()
        features = lines[0].strip().split(",")
        target = features[-1]

        for index, line in enumerate(lines[1:]):
            values = line.strip().split(",")
            dataframe.append(dict())
            for i, feature in enumerate(features):
                dataframe[index][feature] = values[i]

        return target
