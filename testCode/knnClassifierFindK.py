from pprint import pprint as pp
import json
import math
import numpy as np
from numpy import *
from collections import Counter



class knnClassifierFindK:
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

    def returnBestK(self, nearestNeighbors, noOfTestDataSet):
        testDataIndecies = np.unique((nearestNeighbors[:,0]).astype(int))
        testSetLabels = self.metadata[:, -1]
        accuracy = []
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
            accuracy.append([k,(count/noOfTestDataSet * 100)])
        npAccuracy = np.array(accuracy)
        temp = npAccuracy[npAccuracy[:, 1].argsort()]
        temp = temp[temp[:,1]==npAccuracy.max(axis=0)[1]]
        bestK = temp[0][0]
        return bestK

    def predictBestKHyperparameterNew(self, nearestNeighbors, noOfTestDataSet):
        testDataIndecies = np.unique((nearestNeighbors[:, 0]).astype(int))
        testSetLabels = self.metadata[:, -1]
        accuracy = 0.0
        bestK = 1
        for k in range(1, self.k + 1):
            predictedLabels = []
            neighbors = nearestNeighbors.copy()
            for i in testDataIndecies:
                temp = neighbors[neighbors[:, 0] == i]
                temp = temp[temp[:, 1].argsort()]
                kNearestNeighbors = (temp[:k]).astype(int)
                label = (Counter(kNearestNeighbors[:, 2]).most_common(1))[0][0]
                predictedLabels.append(label)
            count = 0
            for index, label in enumerate(predictedLabels):
                if (label == testSetLabels[index]):
                    count += 1
            curAccuracy = count / noOfTestDataSet * 100
            if (curAccuracy > accuracy):
                accuracy = curAccuracy
            elif (curAccuracy == accuracy):
                bestK = k - 1
                break
            else:
                bestK = k
                break
        pp(bestK)
        return bestK

    def predictBestKHyperparameter(self, nearestNeighbors, noOfTestDataSet):
        testDataIndecies = np.unique((nearestNeighbors[:,0]).astype(int))
        testSetLabels = self.metadata[:, -1]

        accuracy = []
        for k in range(1, self.k + 1):
            predictedLabels = []
            # neighbors = nearestNeighbors.copy()
            for i in testDataIndecies:
                temp = nearestNeighbors[nearestNeighbors[:, 0] == i]
                temp = temp[temp[:, 1].argsort()]
                kNearestNeighbors = (temp[:k]).astype(int)
                label = (Counter(kNearestNeighbors[:, 2]).most_common(1))[0][0]
                predictedLabels.append(label)
            count = 0
            for index, label in enumerate(predictedLabels):
                if (label == testSetLabels[index]):
                    count += 1
            acc = (count / noOfTestDataSet)
            print(k, end=",")
            pp(acc)
            accuracy.append([k, acc])
        npAccuracy = np.array(accuracy)
        temp = npAccuracy[npAccuracy[:, 1].argsort()]
        temp = temp[temp[:, 1] == npAccuracy.max(axis=0)[1]]
        bestK = temp[0][0]
        pp(bestK)
        pp(npAccuracy.max(axis=0)[1])
        return bestK


if __name__ == '__main__':

    knn = knnClassifierFindK(3, "./Resources/digits_train.json")
    validateKnn = knnClassifierFindK(3, "./Resources/digits_val.json")

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
    pp("Best K : {}".format(bestK))



    # pp(predictions)
    # ind = np.argpartition(sortedMatrix, knn.k, axis=1)[:, :knn.k]
    # pp(sortedMatrix[:,0:knn.k])
    # # predictions = stdTrainSet[ind, 2].mean(1)
    #
    # pp(predictions)
    # pp("############## Time: {}".format(time.time()))
    # predictedLabels = []
    # for testIndex in range(0, noOfTestDataSet):
    #     for trainIndex in range(0, noOfTrainDataSet):
    #         nearestNeighbors.append([testIndex, (np.abs(stdTrainSet[trainIndex,0:-1]-stdTestSet[testIndex,0:-1])).sum(), stdTrainSet[trainIndex,-1]])
    #         # nearestNeighbors.append([(np.abs(stdTrainSet[trainIndex,0:-1]-stdTestSet[testIndex,0:-1])).sum(), stdTrainSet[trainIndex,-1]])
    #         # nearestNeighbors = np.vstack([nearestNeighbors, [testIndex, (np.abs(stdTrainSet[trainIndex,0:-1]-stdTestSet[testIndex,0:-1])).sum(), stdTrainSet[trainIndex,-1]]])

    # pp(np.array(nearestNeighbors).shape)
    # pp("############## Time: {}".format(time.time()))
    #
    # print(type(nearestNeighbors))
    # nearestNeighbors = np.array(nearestNeighbors)
    # print(nearestNeighbors[0])
    # pp(nearestNeighbors[0:1124,:].shape)



    # for k in range(1, knn.k + 1):
    #     kNearestNeighbors = (temp[:k]).astype(int)
    #     label = (Counter(kNearestNeighbors[:, 1]).most_common(1))[0][0]
    #     predictedLabels.append(label)
    #     count = 0
    #     for index, label in enumerate(predictedLabels):
    #         if (label == valSetLabels[index]):
    #             count += 1
    #     print(k, end=",")
    #     pp(count/noOfTestDataSet)
    #     accuracy.append([k,count/noOfTestDataSet])
    #
    # npAccuracy = np.array(accuracy)
    # temp = npAccuracy[npAccuracy[:, 1].argsort()]
    # temp = temp[temp[:, 1] == npAccuracy.max(axis=0)[1]]
    # bestK = temp[0][0]
    # pp(bestK)
    # pp(npAccuracy.max(axis=0)[1])
    # # bestK = validateKnn.predictBestKHyperparameter(np.array(nearestNeighbors), noOfTestDataSet)
    # # pp(bestK)

    # pp("############## Time: {}".format(time.time()))
    # def euclidean_distance(X_train, X_test):
    #     return [np.linalg.norm(X - X_test) for X in X_train]
    #
    #
    # def k_nearest(X, Y, k):
    #     idx = np.argpartition(X, k)
    #     return np.take(Y, idx[:k])
    #
    #
    # def predict(X_test):
    #     distance_list = [euclidean_distance(stdTrainSet, X) for X in X_test]
    #     return np.array([Counter(k_nearest(distances, stdTrainSet, 3)).most_common()[0][0] for distances in distance_list])
    #
    # result = predict(stdTestSet)
    # pp("############## Tine: {}".format(time.time()))

