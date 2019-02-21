from pprint import pprint as pp
import json
import math
import numpy as np
from numpy import *
from collections import Counter
import time
from scipy.stats import mode
import pandas as pd



class HyperParamKnn:
    def __init__(self, k , inputFilePath):
        self.k = k
        self.inputFile = inputFilePath
        self.features = ""
        self.metadata = ""
        self.labelTypes = []
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
    # knn = HyperParamKnn(k, "./Resources/digits_train.json")
    # validateKnn = HyperParamKnn(k, "./Resources/digits_val.json")
    # testKnn = HyperParamKnn(k, "./Resources/digits_test.json")

    knn = HyperParamKnn(k, "./Resources/votes_train.json")
    validateKnn = HyperParamKnn(k, "./Resources/votes_val.json")
    testKnn = HyperParamKnn(k, "./Resources/votes_test.json")

    args = 9

    digitsTrainingSet = json.load(open(knn.inputFile))
    knn.features = np.array(digitsTrainingSet["metadata"]["features"])
    knn.metadata = np.array(digitsTrainingSet["data"])
    knn.labelTypes = knn.features[-1][1]
    knn.labels = knn.metadata[:, -1]
    # knn.mean = np.mean(knn.metadata,axis=0)
    # knn.sd = np.std(knn.metadata, axis=0)
    # knn.sd[knn.sd == 0.00] = 1.0

    # noOfTrainDataSet = stdTrainSet.shape[0]
    # noOfTestDataSet = stdTestSet.shape[0]
    # noOfValDataSet = stdValSet.shape[0]

    # pp(np.abs(stdTestSet[:, None, 0:-1] - stdTrainSet[None, :, 0:-1])[0][0])
    # sortedMatrix = np.sort(distanceMatrix, 1)[:, 2:, ]
    if(args == -2):
        digitsTestSet = json.load(open(testKnn.inputFile))
        testKnn.features = np.array(digitsTestSet["metadata"]["features"])
        testKnn.metadata = np.array(digitsTestSet["data"])
        testKnn.labels = testKnn.metadata[:, -1]
        noOfTestDataSet = testKnn.metadata.shape[0]


        batchSize = knn.metadata.shape[0] / 10
        for i in range(1, 11):
            sliceSize = int(batchSize * i)

            distanceMatrix = np.sum(testKnn.metadata[:, None, 0:-1] != knn.metadata[None, 0:sliceSize, 0:-1], -1)
            ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
            predictions = knn.metadata[0:sliceSize,:][ind, -1]
            predictions = np.sort(predictions, axis=1)
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(testKnn.labels))
            res = count.item() / noOfTestDataSet
            print(sliceSize, end=",")
            print(res)

    elif (args == -1):

        digitsValSet = json.load(open(validateKnn.inputFile))
        validateKnn.features = np.array(digitsValSet["metadata"]["features"])
        validateKnn.metadata = np.array(digitsValSet["data"])
        validateKnn.labels = validateKnn.metadata[:, -1]

        noOfValDataSet = validateKnn.metadata.shape[0]

        accuracy = []
        distanceMatrix = np.sum(np.absolute(validateKnn.metadata[:, None, 0:-1] != knn.metadata[None, :, 0:-1]), -1)
        for kValue in range(1, knn.k+1):
            #ind = np.argpartition(distanceMatrix, kValue, axis=1)[:, :kValue]
            ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :kValue]
            predictions = knn.metadata[ind, -1]
            result = np.array(mode(predictions, axis=1)[0])
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(validateKnn.labels))
            res = count.item() / noOfValDataSet
            print(kValue, end=",")
            print(res)
            accuracy.append([kValue, res])

        npAccuracy = np.array(accuracy)
        temp = npAccuracy[npAccuracy[:, 1].argsort()]
        temp = temp[temp[:, 1] == npAccuracy.max(axis=0)[1]]
        temp = temp[temp[:, 0].argsort()]
        bestK = int(temp[0][0])
        print(bestK)
        # pp(npAccuracy.max(axis=0)[1])

        newK = bestK

        digitsTestSet = json.load(open(testKnn.inputFile))
        testKnn.features = np.array(digitsTestSet["metadata"]["features"])
        testKnn.metadata = np.array(digitsTestSet["data"])
        testKnn.labels = testKnn.metadata[:, -1]

        # Combine Train and Validation dataSet
        trainValMetadata = np.concatenate((knn.metadata, validateKnn.metadata))
        noOfTestDataSet = testKnn.metadata.shape[0]

        distanceMatrix = np.sum(np.absolute(testKnn.metadata[:, None, 0:-1] != trainValMetadata[None, :, 0:-1]), -1)
        #ind = np.argpartition(distanceMatrix, newK, axis=1)[:, :newK]
        ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :newK]
        predictions = trainValMetadata[ind, -1]
        predictions = np.sort(predictions, axis=1)

        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(testKnn.labels))
        res = count.item() / noOfTestDataSet
        print(res)
    elif (args == 0):

        digitsTestSet = json.load(open(testKnn.inputFile))
        testKnn.features = np.array(digitsTestSet["metadata"]["features"])
        testKnn.metadata = np.array(digitsTestSet["data"])
        testKnn.labels = testKnn.metadata[:, -1]

        noOfTestDataSet = testKnn.metadata.shape[0]

        distanceMatrix = np.count_nonzero(testKnn.metadata[:, None, 0:-1] != knn.metadata[None, :, 0:-1], -1)
        ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
        predictions = knn.metadata[ind, -1]
        result = np.array(mode(predictions, axis=1)[0])

        for index in range(predictions.shape[0]):
            mapper = Counter(predictions[index].tolist())
            for i in knn.labelTypes:
                if (mapper.get(i) == None):
                    print(0, end=",")
                else:
                    print(mapper.get(i), end=",")
            print(result[index][0])
        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(testKnn.labels))
        res = count.item() / noOfTestDataSet
        print(k, end=",")
        print(res)
    elif (args == 1):

        # stdTrainSet = knn.standardiseFeatures(np.asfarray(knn.metadata, float))
        stdTrainSet = (np.asfarray(knn.metadata[:, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
        stdTrainSet = np.column_stack([stdTrainSet, knn.metadata[:, -1]])


        digitsTestSet = json.load(open(testKnn.inputFile))
        testKnn.features = np.array(digitsTestSet["metadata"]["features"])
        testKnn.metadata = np.array(digitsTestSet["data"])
        testKnn.labels = testKnn.metadata[:, -1]

        # stdTestSet = knn.standardiseFeatures(np.asfarray(testKnn.metadata, float))
        stdTestSet = (np.asfarray(testKnn.metadata[:, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
        stdTestSet = np.column_stack([stdTestSet, testKnn.metadata[:, -1]])
        noOfTestDataSet = stdTestSet.shape[0]

        distanceMatrix = np.sum(np.absolute(stdTestSet[:, None, 0:-1] - stdTrainSet[None, :, 0:-1]), -1)
        ind = np.argpartition(distanceMatrix, k, axis=1)[:, :k]
        predictions = stdTrainSet[ind, -1].astype(int)
        predictions = np.sort(predictions, axis=1)
        result = np.array(mode(predictions, axis=1)[0])
        for index in range(predictions.shape[0]):
            mapper = Counter(predictions[index].tolist())
            for i in knn.labelTypes:
                if (mapper.get(i) == None):
                    print(0, end=",")
                else:
                    print(mapper.get(i), end=",")
            pp(result[index][0])
        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(testKnn.labels))
        res = count.item() / noOfTestDataSet
        print(k, end=",")
        pp(res)
    elif (args == 2):

        # stdTrainSet = knn.standardiseFeatures(np.asfarray(knn.metadata, float))
        stdTrainSet = (np.asfarray(knn.metadata[:, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
        stdTrainSet = np.column_stack([stdTrainSet, knn.metadata[:, -1]])


        digitsValSet = json.load(open(validateKnn.inputFile))
        validateKnn.features = np.array(digitsValSet["metadata"]["features"])
        validateKnn.metadata = np.array(digitsValSet["data"])
        validateKnn.labels = validateKnn.metadata[:, -1]

        # stdValSet = knn.standardiseFeatures(np.asfarray(validateKnn.metadata, float))
        stdValSet = (np.asfarray(validateKnn.metadata[:, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
        stdValSet = np.column_stack([stdValSet, validateKnn.metadata[:, -1]])
        noOfValDataSet = stdValSet.shape[0]

        accuracy = []
        distanceMatrix = np.sum(np.absolute(stdValSet[:, None, 0:-1] - stdTrainSet[None, :, 0:-1]), -1)
        for kValue in range(1,knn.k):
            ind = np.argpartition(distanceMatrix, kValue, axis=1)[:, :kValue]
            predictions = stdTrainSet[ind, -1].astype(int)
            #predictions  = np.sort(predictions, axis=1)
            result = np.array(mode(predictions, axis=1)[0])
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(validateKnn.labels))
            res = count.item() / noOfValDataSet
            print(kValue, end=",")
            pp(res)
            accuracy.append([kValue, res])

        npAccuracy = np.array(accuracy)
        temp = npAccuracy[npAccuracy[:, 1].argsort()]
        temp = temp[temp[:, 1] == npAccuracy.max(axis=0)[1]]
        bestK = int(temp[0][0])
        pp(bestK)
        #pp(npAccuracy.max(axis=0)[1])

        newK = bestK

        digitsTestSet = json.load(open(testKnn.inputFile))
        testKnn.features = np.array(digitsTestSet["metadata"]["features"])
        testKnn.metadata = np.array(digitsTestSet["data"])
        testKnn.labels = testKnn.metadata[:, -1]

        # Combine Train and Validation dataSet
        trainValMetadata = np.concatenate((knn.metadata,validateKnn.metadata))
        mean = np.mean(trainValMetadata, axis=0)
        sd = np.std(trainValMetadata, axis=0)
        sd[sd == 0.00] = 1.0

        # stdTrainSet = knn.standardiseFeatures(np.asfarray(knn.metadata, float))
        stdTrainValSet = (np.asfarray(trainValMetadata[:, 0:-1], float) - mean[0:-1]) / sd[0:-1]
        stdTrainValSet = np.column_stack([stdTrainValSet, trainValMetadata[:, -1]])

        # stdTestSet = knn.standardiseFeatures(np.asfarray(testKnn.metadata, float))
        stdTestSet = (np.asfarray(testKnn.metadata[:, 0:-1], float) - mean[0:-1]) / sd[0:-1]
        stdTestSet = np.column_stack([stdTestSet, testKnn.metadata[:, -1]])
        noOfTestDataSet = stdTestSet.shape[0]

        distanceMatrix = np.sum(np.absolute(stdTestSet[:, None, 0:-1] - stdTrainValSet[None, :, 0:-1]), -1)
        ind = np.argpartition(distanceMatrix, newK, axis=1)[:, :newK]
        predictions = stdTrainValSet[ind, -1].astype(int)
        predictions = np.sort(predictions, axis=1)

        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(testKnn.labels))
        res = count.item() / noOfTestDataSet
        pp(res)
    elif(args == 3):
        digitsTestSet = json.load(open(testKnn.inputFile))
        testKnn.features = np.array(digitsTestSet["metadata"]["features"])
        testKnn.metadata = np.array(digitsTestSet["data"])
        testKnn.labels = testKnn.metadata[:, -1]
        noOfTestDataSet = testKnn.metadata.shape[0]

        batchSize = knn.metadata.shape[0] / 10
        for i in range(1, 11):
            sliceSize = int(batchSize * i)
            metaData = knn.metadata[0:sliceSize]

            knn.mean = np.mean(metaData, axis=0)
            knn.sd = np.std(metaData, axis=0)
            knn.sd[knn.sd == 0.00] = 1.0

            stdTrainSet = (np.asfarray(knn.metadata[0:sliceSize, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
            stdTrainSet = np.column_stack([stdTrainSet, knn.metadata[0:sliceSize, -1]])

            stdTestSet = (np.asfarray(testKnn.metadata[:, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
            stdTestSet = np.column_stack([stdTestSet, testKnn.metadata[:, -1]])

            distanceMatrix = np.sum(np.absolute(stdTestSet[:, None, 0:-1] - stdTrainSet[None, :, 0:-1]), -1)
            ind = np.argpartition(distanceMatrix, k, axis=1)[:, :k]
            predictions = stdTrainSet[ind, -1].astype(int)
            predictions = np.sort(predictions, axis=1)
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(testKnn.labels))
            res = count.item() / noOfTestDataSet
            print(sliceSize, end=",")
            pp(res)
    else:
        digitsTestSet = json.load(open(testKnn.inputFile))
        testKnn.features = np.array(digitsTestSet["metadata"]["features"])
        testKnn.metadata = np.array(digitsTestSet["data"])
        testKnn.labels = testKnn.metadata[:, -1]
        noOfTestDataSet = testKnn.metadata.shape[0]

        category = testKnn.features[-1][1]
        if (len(category) != 2):
            print("ROC can be drawn only on caterogical dataset, please check youe dataset and label types")
            exit(1)
        else:
            labelMap = {}
            labelMap[category[0]] = 1
            labelMap[category[1]] = 0

            trainigLabels = knn.metadata[:,-1].copy()
            for index,label in enumerate(trainigLabels):
                trainigLabels[index,] = labelMap.get(label)
            trainigLabels = trainigLabels.astype(int)

            testLabels = testKnn.metadata[:, -1].copy()
            for index, label in enumerate(testLabels):
                testLabels[index,] = labelMap.get(label)
            testLabels = testLabels.astype(int)

            constant = math.pow(10,-5)

            distanceMatrix = np.count_nonzero(testKnn.metadata[:, None, 0:-1] != knn.metadata[None, :, 0:-1], -1)

            ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
            predictedLabels = knn.metadata[ind, -1]
            predictedLabels = (predictedLabels==category[0])*1

            distanceMatrix = np.sort(distanceMatrix, axis=1)[:,:k]
            weightedDistance = 1 / (np.square(distanceMatrix) + constant)
            weightedSum = np.sum(weightedDistance, axis=1)
            confidence = np.sum(weightedDistance * predictedLabels,axis=1)/weightedSum
            rocMatrix = np.column_stack((testLabels,confidence))
            rocMatrixSorted = rocMatrix[np.argsort([rocMatrix[:, 1]])][0]
            posNegCount = Counter(rocMatrixSorted[:,0].astype(int))
            posCount = posNegCount.get(1)
            negCount = posNegCount.get(0)

            TP = 0
            FP = 0
            last_TP = 0
            for i in range(testKnn.metadata.shape[0]):
                if (i > 0) and (rocMatrixSorted[i,1] != rocMatrixSorted[i-1,1]) and ( testLabels[i] < 1) and ( TP > last_TP):
                    FPR = FP / negCount
                    TPR = TP / posCount
                    print(FPR, end=",")
                    print(TPR)
                    last_TP = TP
                if testLabels[i] > 0:
                    TP += 1
                else:
                    FP += 1
            FPR = FP / negCount
            TPR = TP / posCount
            print(FPR, end=",")
            print(TPR)