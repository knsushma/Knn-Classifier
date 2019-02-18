from pprint import pprint as pp
import json
import math
import numpy as np
from numpy import *
from collections import Counter



class KnnClassifier:
    def __init__(self, k , inputFilePath):
        self.k = k
        self.inputFile = inputFilePath
        self.features = ""
        self.metadata = ""
        self.labels = []
        self.mean = []
        self.sd = []

    def standardiseFeatures(self, dataSet):
        for i in range(0, len(self.features)-1):
            dataSet[:,i] = dataSet[:,i] - self.mean[i]
            dataSet[:,i] = dataSet[:,i] / self.sd[i]
        return dataSet

    def predictBestKHyperparameter(self, nearestNeighbors, predictedLabels):
        #  nearestNeighbors[nearestNeighbors[:,0]==0][:,[1,2]]
        # nearestNeighbors[np.ix_(nearestNeighbors[:, 0] > 3, (0, 1))]
        # nearestNeighbors[nearestNeighbors[:, 0] > 3][:, np.array([False, True])]
        nearestNeighbors.sort(key=lambda x: x[1])
        kNearestNeighbors = np.array(nearestNeighbors[:self.k]).astype(int)
        pp(kNearestNeighbors[:, 2])
        pp((Counter(kNearestNeighbors[:, 2]).most_common(1)[0][0]))
        label = (Counter(kNearestNeighbors[:, 2]).most_common(1))[0][0]
        mapper = Counter(kNearestNeighbors[:, 2])
        for i in knn.labels:
            if (mapper.get(i) == None):
                print(0, end=",")
            else:
                print(mapper.get(i), end=",")
        print(label)
        predictedLabels.append(label)
        return predictedLabels

    def printKNearestNeighbors(self, nearestNeighbors, predictedLabels):
        nearestNeighbors.sort(key=lambda x: x[0])
        kNearestNeighbors = np.array(nearestNeighbors[:self.k]).astype(int)
        label = (Counter(kNearestNeighbors[:, 1]).most_common(1))[0][0]

        mapper = Counter(kNearestNeighbors[:, 1])
        for i in knn.labels:
            if (mapper.get(i) == None):
                print(0, end=",")
            else:
                print(mapper.get(i), end=",")
        print(label)
        predictedLabels.append(label)
        return predictedLabels

    def printAccuracyOfModel(self, predictedLabel, noOfTestSet):
        count = 0
        testSetLabels = self.metadata[:, -1]
        for index, label in enumerate(predictedLabel):
            if (label == testSetLabels[index]):
                count += 1
        pp("Accuracy: {}".format(count / noOfTestSet * 100))



if __name__ == '__main__':

    knn = KnnClassifier(10, "./Resources/digits_train.json")
    testKnn = KnnClassifier(3, "./Resources/digits_test.json")

    digitsTrainingSet = json.load(open(knn.inputFile))
    knn.features = np.array(digitsTrainingSet["metadata"]["features"])
    knn.metadata = np.array(digitsTrainingSet["data"])
    knn.labels = knn.features[-1][1]

    noOfFeatures = len(knn.features)-1
    knn.mean = np.mean(knn.metadata,axis=0)
    knn.sd = np.std(knn.metadata, axis=0)


    knn.sd[knn.sd == 0.00] = 1.0
    stdTrainSet = knn.standardiseFeatures(np.asfarray(knn.metadata, float))

    digitsTestSet = json.load(open(testKnn.inputFile))
    testKnn.features = np.array(digitsTestSet["metadata"]["features"])
    testKnn.metadata = np.array(digitsTestSet["data"])
    stdTestSet = knn.standardiseFeatures(np.asfarray(testKnn.metadata, float))

    noOfTestDataSet = stdTestSet.shape[0]
    noOfTrainDataSet = stdTrainSet.shape[0]

    predictedLabel = []
    for testIndex in range(0, noOfTestDataSet):
        nearestNeighbors = []
        for trainIndex in range(0, noOfTrainDataSet):
            nearestNeighbors.append([(np.abs(stdTrainSet[trainIndex,0:-1]-stdTestSet[testIndex,0:-1])).sum(), stdTrainSet[trainIndex,-1]])
        predictedLabel = knn.printKNearestNeighbors(nearestNeighbors, predictedLabel)
    testKnn.printAccuracyOfModel(predictedLabel, noOfTestDataSet)