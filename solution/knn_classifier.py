import json
import numpy as np
from collections import Counter
import sys


class knnClassifier:
    def __init__(self, k , inputFilePath):
        self.k = k
        self.inputFile = inputFilePath
        self.features = []
        self.metadata = [0][0]
        self.labelTypes = []
        self.labels = []
        self.mean = []
        self.sd = []
        self.shape = (0, 0)
        self.sliceSize = 0


    def loadAndInitDataSet(self, inputFileName):
        dataSet = json.load(open(inputFileName))
        self.features = np.array(dataSet["metadata"]["features"])
        self.metadata = np.array(dataSet["data"])
        self.labels = self.metadata[:, -1]
        self.labelTypes = self.features[-1][1]
        self.shape = self.metadata.shape
        self.sliceSize = self.metadata.shape[0]

    def computeMeanAndSD(self, dataSet):
        self.mean = np.mean(dataSet, axis=0)
        self.sd = np.std(dataSet, axis=0)
        self.sd[self.sd == 0.00] = 1.0

    def findDistanceMatrix(self, dataSetObj):
        manhattanIndices = []
        hammingIndices = []
        for index in range(len(self.features) - 1):
            if (self.features[index][1] == "numeric"):
                manhattanIndices.append(index)
            else:
                hammingIndices.append(index)

        manhattanDistMatrix = np.zeros((dataSetObj.shape[0],len(manhattanIndices)))
        if(len(manhattanIndices)):
            trainMatrix = self.metadata[:,manhattanIndices]
            testMatrix = dataSetObj.metadata[:,manhattanIndices]
            self.computeMeanAndSD(trainMatrix[0:self.sliceSize])
            stdTrainSet = (trainMatrix - self.mean) / self.sd
            stdTestSet = (testMatrix - self.mean) / self.sd
            manhattanDistMatrix = np.sum(np.absolute(stdTestSet[:, None, :] - stdTrainSet[None, 0:self.sliceSize, :]), -1)

        hammingDistMatrix = np.zeros((dataSetObj.shape[0], len(hammingIndices)))
        if(len(hammingIndices)):
            trainMatrix = self.metadata[:, hammingIndices]
            testMatrix = dataSetObj.metadata[:, hammingIndices]
            hammingDistMatrix = np.count_nonzero(testMatrix[:, None, :] != trainMatrix[None, 0:self.sliceSize, :], -1)

        if(len(manhattanIndices) and len(hammingIndices)):
            return manhattanDistMatrix+hammingDistMatrix
        elif(len(manhattanIndices)):
            return manhattanDistMatrix
        else:
            return hammingDistMatrix

if __name__ == '__main__':

    if (len(sys.argv)<3):
        print("Please pass 3 arguments. 1) K value to compute KNN, 2) Traning File path, 3) Testing File path")
        sys.exit(1)

    k = int(sys.argv[1])
    knn = knnClassifier(k, sys.argv[2])
    testKnn = knnClassifier(k, sys.argv[3])

    knn.loadAndInitDataSet(knn.inputFile)
    testKnn.loadAndInitDataSet(testKnn.inputFile)

    distanceMatrix = knn.findDistanceMatrix(testKnn)
    ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
    predictions = knn.metadata[ind, -1]
    predictions = np.sort(predictions, axis=1)
    #predictedLabels = np.array(mode(predictions, axis=1)[0])
    repeats = []
    for i in range(predictions.shape[0]):
        repeats.append(Counter(predictions[i]).most_common(1)[0])
    predictedLabels = np.array(repeats)
    for index in range(predictions.shape[0]):
        mapper = Counter(predictions[index].tolist())
        for i in knn.labelTypes:
            if (mapper.get(i) == None):
                print(0, end=",")
            else:
                print(mapper.get(i), end=",")
        print(predictedLabels[index][0])