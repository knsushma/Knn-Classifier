from pprint import pprint as pp
import json
import math
import numpy as np
from numpy import *
from collections import Counter



class HyperParamKnn:
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

    def predictBestKHyperparameter(self, nearestNeighbors, noOfTestDataSet):

        testDataIndecies = np.unique((nearestNeighbors[:,0]).astype(int))
        testSetLabels = self.metadata[:, -1]

        accuracy = 0.0
        bestK = 1
        for k in range(1,self.k+1):
            predictedLabels = []
            neighbors = nearestNeighbors.copy()
            for i in testDataIndecies:
                temp = neighbors[neighbors[:,0]==i]
                temp = temp[temp[:, 1].argsort()]
                kNearestNeighbors = (temp[:k]).astype(int)
                label = (Counter(kNearestNeighbors[:, 2]).most_common(1))[0][0]
                predictedLabels.append(label)
            count = 0
            for index, label in enumerate(predictedLabels):
                if (label == testSetLabels[index]):
                    count += 1
            curAccuracy = count/noOfTestDataSet * 100
            if (curAccuracy > accuracy):
                accuracy = curAccuracy
            elif (curAccuracy == accuracy):
                bestK = k-1
                break
            else:
                bestK = k
                break

        pp(bestK)
        #     accuracy.append([k,(count/noOfTestDataSet * 100)])
        # npAccuracy = np.array(accuracy)
        # temp = npAccuracy[npAccuracy[:, 1].argsort()]
        # temp = temp[temp[:,1]==npAccuracy.max(axis=0)[1]]
        # bestK = temp[0][0]
        return bestK


if __name__ == '__main__':

    knn = HyperParamKnn(10, "./Resources/digits_train.json")
    validateKnn = HyperParamKnn(10, "./Resources/digits_val.json")

    digitsTrainingSet = json.load(open(knn.inputFile))
    knn.features = np.array(digitsTrainingSet["metadata"]["features"])
    knn.metadata = np.array(digitsTrainingSet["data"])
    knn.labels = knn.features[-1][1]

    noOfFeatures = len(knn.features)-1
    knn.mean = np.mean(knn.metadata,axis=0)
    knn.sd = np.std(knn.metadata, axis=0)


    knn.sd[knn.sd == 0.00] = 1.0
    stdTrainSet = knn.standardiseFeatures(np.asfarray(knn.metadata, float))

    digitsTestSet = json.load(open(validateKnn.inputFile))
    validateKnn.features = np.array(digitsTestSet["metadata"]["features"])
    validateKnn.metadata = np.array(digitsTestSet["data"])
    stdTestSet = knn.standardiseFeatures(np.asfarray(validateKnn.metadata, float))

    noOfTestDataSet = stdTestSet.shape[0]
    noOfTrainDataSet = stdTrainSet.shape[0]

    #noOfTestDataSet = 20
    predictedLabel = []
    nearestNeighbors = []
    for testIndex in range(0, noOfTestDataSet):
        for trainIndex in range(0, noOfTrainDataSet):
            nearestNeighbors.append([testIndex, (np.abs(stdTrainSet[trainIndex,0:-1]-stdTestSet[testIndex,0:-1])).sum(), stdTrainSet[trainIndex,-1]])
    bestK = validateKnn.predictBestKHyperparameter(np.array(nearestNeighbors), noOfTestDataSet)
    pp(bestK)
