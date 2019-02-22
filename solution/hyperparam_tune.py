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

    if (len(sys.argv)<4):
        print("Please pass 4 arguments. 1) K value to compute KNN, 2) Traning File path, 3) Validation File path, 4) Testing File Path")
        sys.exit(1)

    k = int(sys.argv[1])
    knn = knnClassifier(k, sys.argv[2])
    validateKnn = knnClassifier(k, sys.argv[3])
    testKnn = knnClassifier(k, sys.argv[4])

    knn.loadAndInitDataSet(knn.inputFile)
    validateKnn.loadAndInitDataSet(validateKnn.inputFile)
    accuracyList = []
    distanceMatrix = knn.findDistanceMatrix(validateKnn)
    for kValue in range(1,knn.k+1):
        ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :kValue]
        predictions = knn.metadata[ind, -1]
        predictions  = np.sort(predictions, axis=1)
        #predictedLabels = np.array(mode(predictions, axis=1)[0]).flatten()
        repeats = []
        for i in range(predictions.shape[0]):
            repeats.append(Counter(predictions[i]).most_common(1)[0][0])
        predictedLabels = np.array(repeats)
        count = np.sum(predictedLabels == np.array(validateKnn.labels))
        accuracy = count.item() / validateKnn.shape[0]
        print(kValue, end=",")
        print(accuracy)
        accuracyList.append([kValue, accuracy])

    npAccuracy = np.array(accuracyList)
    temp = npAccuracy[npAccuracy[:, 1].argsort()]
    temp = temp[temp[:, 1] == npAccuracy.max(axis=0)[1]]
    temp = temp[temp[:, 0].argsort()]
    bestK = int(temp[0][0])
    print(bestK)

    testKnn.loadAndInitDataSet(testKnn.inputFile)
    trainValMetadata = np.concatenate((knn.metadata,validateKnn.metadata))
    knn.metadata = trainValMetadata
    knn.sliceSize = knn.metadata.shape[0]
    distanceMatrix = knn.findDistanceMatrix(testKnn)
    ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :bestK]
    predictions = knn.metadata[ind, -1]
    predictions = np.sort(predictions, axis=1)
    # flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
    repeats = []
    for i in range(predictions.shape[0]):
        repeats.append(Counter(predictions[i]).most_common(1)[0][0])
    flatResult = np.array(repeats).flatten()
    count = np.sum(flatResult == np.array(testKnn.labels))
    accuracy = count.item() / testKnn.shape[0]
    print(accuracy)