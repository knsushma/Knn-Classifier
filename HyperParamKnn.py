from pprint import pprint as pp
import json
import math
import numpy as np
from numpy import *
from collections import Counter
import time
from scipy.stats import mode



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

    def predictBestKHyperparameterOld(self, nearestNeighbors, noOfTestDataSet):
        testDataIndecies = np.unique((nearestNeighbors[:,0]).astype(int))
        testSetLabels = self.metadata[:, -1]
        accuracy = []
        for k in range(1,self.k+1):
            predictedLabels = []
            # neighbors = nearestNeighbors.copy()
            for i in testDataIndecies:
                temp = nearestNeighbors[nearestNeighbors[:,0]==i]
                temp = temp[temp[:, 1].argsort()]
                kNearestNeighbors = (temp[:k]).astype(int)
                label = (Counter(kNearestNeighbors[:, 2]).most_common(1))[0][0]
                predictedLabels.append(label)
            count = 0
            for index, label in enumerate(predictedLabels):
                if (label == testSetLabels[index]):
                    count += 1
            acc = (count / noOfTestDataSet)
            print(k, end = ",")
            pp(acc)
            accuracy.append([k, acc])
        npAccuracy = np.array(accuracy)
        temp = npAccuracy[npAccuracy[:, 1].argsort()]
        temp = temp[temp[:, 1] == npAccuracy.max(axis=0)[1]]
        bestK = temp[0][0]
        pp(bestK)
        pp(npAccuracy.max(axis=0)[1])
        return bestK

    def predictBestKHyperparameter(self, nearestNeighbors, predictedLabels):
        nearestNeighbors.sort(key=lambda x: x[0])
        kNearestNeighbors = np.array(nearestNeighbors[:self.k]).astype(int)
        label = (Counter(kNearestNeighbors[:, 1]).most_common(1))[0][0]
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

    k = 20
    knn = HyperParamKnn(k, "./Resources/digits_train.json")
    validateKnn = HyperParamKnn(k, "./Resources/digits_test.json")

    digitsTrainingSet = json.load(open(knn.inputFile))
    knn.features = np.array(digitsTrainingSet["metadata"]["features"])
    knn.metadata = np.array(digitsTrainingSet["data"])
    knn.labels = knn.features[-1][1]

    noOfFeatures = len(knn.features)-1
    knn.mean = np.mean(knn.metadata,axis=0)
    knn.sd = np.std(knn.metadata, axis=0)
    knn.sd[knn.sd == 0.00] = 1.0

    #stdTrainSet = knn.standardiseFeatures(np.asfarray(knn.metadata, float))
    stdTrainSet = (np.asfarray(knn.metadata[:,0:-1], float) - knn.mean[0:-1])/knn.sd[0:-1]
    stdTrainSet = np.column_stack([stdTrainSet,knn.metadata[:,-1]])

    digitsTestSet = json.load(open(validateKnn.inputFile))
    validateKnn.features = np.array(digitsTestSet["metadata"]["features"])
    validateKnn.metadata = np.array(digitsTestSet["data"])
    validateKnn.labels = validateKnn.metadata[:, -1]
    # stdValSet = knn.standardiseFeatures(np.asfarray(validateKnn.metadata, float))
    stdValSet = (np.asfarray(validateKnn.metadata[:, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
    stdValSet = np.column_stack([stdValSet, validateKnn.metadata[:, -1]])

    noOfValDataSet = stdValSet.shape[0]
    noOfTrainDataSet = stdTrainSet.shape[0]

    # pp(np.abs(stdTestSet[:, None, 0:-1] - stdTrainSet[None, :, 0:-1])[0][0])
    # sortedMatrix = np.sort(distanceMatrix, 1)[:, 2:, ]
    distanceMatrix = np.sum(np.absolute(stdValSet[:, None, 0:-1] - stdTrainSet[None, :, 0:-1]), -1)
    for kValue in range(knn.k+1):
        ind = np.argpartition(distanceMatrix, kValue, axis=1)[:, :kValue]
        predictions = stdTrainSet[ind, -1].astype(int)
        #predictions  = np.sort(predictions, axis=1)

        result = np.array(mode(predictions, axis=1)[0])

        # for index in range(predictions.shape[0]):
        #     mapper = Counter(predictions[index].tolist())
        #     for i in knn.labels:
        #         if (mapper.get(i) == None):
        #             print(0, end=",")
        #         else:
        #             print(mapper.get(i), end=",")
        #     pp(result[index][0])
        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(validateKnn.labels))
        res = count.item() / noOfValDataSet
        print(kValue, end=",")
        pp(res)









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


