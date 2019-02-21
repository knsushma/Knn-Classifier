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
        self.features = []
        self.metadata = [0][0]
        self.labelTypes = []
        self.labels = []
        self.mean = []
        self.sd = []
        self.shape = (0, 0)


    def loadAndInitDataSet(self, inputFileName):
        dataSet = json.load(open(inputFileName))
        self.features = np.array(dataSet["metadata"]["features"])
        self.metadata = np.array(dataSet["data"])
        self.labels = self.metadata[:, -1]
        self.labelTypes = self.features[-1][1]
        self.shape = self.metadata.shape

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
            pp("Inside First IF")
            trainMatrix = self.metadata[:,manhattanIndices]
            testMatrix = dataSetObj.metadata[:,manhattanIndices]
            self.computeMeanAndSD(trainMatrix)
            stdTrainSet = (trainMatrix - self.mean) / self.sd
            stdTestSet = (testMatrix - self.mean) / self.sd
            manhattanDistMatrix = np.sum(np.absolute(stdTestSet[:, None, :] - stdTrainSet[None, :, :]), -1)

        hammingDistMatrix = np.zeros((dataSetObj.shape[0], len(hammingIndices)))
        if(len(hammingIndices)):
            pp("Inside Second IF")
            trainMatrix = self.metadata[:, hammingIndices]
            testMatrix = dataSetObj.metadata[:, hammingIndices]
            hammingDistMatrix = np.count_nonzero(testMatrix[:, None, :] != trainMatrix[None, :, :], -1)

        if(len(manhattanIndices) and len(hammingIndices)):
            return np.concatenate((manhattanDistMatrix, hammingDistMatrix))
        elif(len(manhattanIndices)):
            return manhattanDistMatrix
        else:
            pp("Inside else")
            return hammingDistMatrix

if __name__ == '__main__':

    k = 10
    # knn = HyperParamKnn(k, "./Resources/digits_train.json")
    # validateKnn = HyperParamKnn(k, "./Resources/digits_val.json")
    # testKnn = HyperParamKnn(k, "./Resources/digits_test.json")

    knn = HyperParamKnn(k, "./Resources/votes_train.json")
    validateKnn = HyperParamKnn(k, "./Resources/votes_val.json")
    testKnn = HyperParamKnn(k, "./Resources/votes_test.json")

    args = 1

    knn.loadAndInitDataSet(knn.inputFile)
    # knn.mean = np.mean(knn.metadata,axis=0)
    # knn.sd = np.std(knn.metadata, axis=0)
    # knn.sd[knn.sd == 0.00] = 1.0

    if (args == 1):

        testKnn.loadAndInitDataSet(testKnn.inputFile)

        # knn.computeMeanAndSD(knn.metadata)
        # stdTrainSet = (knn.metadata[:, 0:-1] - knn.mean[0:-1]) / knn.sd[0:-1]
        #
        # testKnn.loadAndInitDataSet(testKnn.inputFile)
        # stdTestSet = (testKnn.metadata[:, 0:-1] - knn.mean[0:-1]) / knn.sd[0:-1]

        # stdTrainSet = knn.metadata[:,0:-1]
        # stdTestSet = testKnn.metadata[:,0:-1]
        #
        # distanceMatrix = np.sum(np.absolute(stdTestSet[:, None, :] != stdTrainSet[None, :, :]), -1)
        distanceMatrix = knn.findDistanceMatrix(testKnn)
        ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
        predictions = knn.metadata[ind, -1]
        predictions = np.sort(predictions, axis=1)
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
        accuracy = count.item() / testKnn.shape[0]
        print(k, end=",")
        print(accuracy)
    elif (args == 2):
        knn.computeMeanAndSD(knn.metadata)
        stdTrainSet = (np.asfarray(knn.metadata[:, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
        stdTrainSet = np.column_stack([stdTrainSet, knn.metadata[:, -1]])

        validateKnn.loadAndInitDataSet(validateKnn.inputFile)
        stdValSet = (np.asfarray(validateKnn.metadata[:, 0:-1], float) - knn.mean[0:-1]) / knn.sd[0:-1]
        stdValSet = np.column_stack([stdValSet, validateKnn.metadata[:, -1]])

        accuracyList = []
        distanceMatrix = np.sum(np.absolute(stdValSet[:, None, 0:-1] - stdTrainSet[None, :, 0:-1]), -1)
        for kValue in range(1,knn.k):
            ind = np.argpartition(distanceMatrix, kValue, axis=1)[:, :kValue]
            predictions = stdTrainSet[ind, -1].astype(int)
            predictions  = np.sort(predictions, axis=1)
            result = np.array(mode(predictions, axis=1)[0])
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(validateKnn.labels))
            accuracy = count.item() / stdValSet.shape[0]
            print(kValue, end=",")
            print(accuracy)
            accuracyList.append([kValue, accuracy])

        npAccuracy = np.array(accuracyList)
        temp = npAccuracy[npAccuracy[:, 1].argsort()]
        temp = temp[temp[:, 1] == npAccuracy.max(axis=0)[1]]
        bestK = int(temp[0][0])
        pp(bestK)

        newK = bestK

        testKnn.loadAndInitDataSet(testKnn.inputFile)

        # Combine Train and Validation dataSet
        trainValMetadata = np.concatenate((knn.metadata,validateKnn.metadata))
        mean = np.mean(trainValMetadata, axis=0)
        sd = np.std(trainValMetadata, axis=0)
        sd[sd == 0.00] = 1.0

        stdTrainValSet = (np.asfarray(trainValMetadata[:, 0:-1], float) - mean[0:-1]) / sd[0:-1]
        stdTrainValSet = np.column_stack([stdTrainValSet, trainValMetadata[:, -1]])

        stdTestSet = (np.asfarray(testKnn.metadata[:, 0:-1], float) - mean[0:-1]) / sd[0:-1]
        stdTestSet = np.column_stack([stdTestSet, testKnn.metadata[:, -1]])

        distanceMatrix = np.sum(np.absolute(stdTestSet[:, None, 0:-1] - stdTrainValSet[None, :, 0:-1]), -1)
        ind = np.argpartition(distanceMatrix, newK, axis=1)[:, :newK]
        predictions = stdTrainValSet[ind, -1].astype(int)
        predictions = np.sort(predictions, axis=1)
        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(testKnn.labels))
        accuracy = count.item() / stdTestSet.shape[0]
        print(accuracy)
    elif(args == 3):
        testKnn.loadAndInitDataSet(testKnn.inputFile)

        batchSize = knn.shape[0] / 10
        for i in range(1, 11):
            sliceSize = int(batchSize * i)
            metaData = knn.metadata[0:sliceSize]

            newMean = np.mean(metaData, axis=0)
            newSD = np.std(metaData, axis=0)
            newSD[newSD == 0.00] = 1.0

            stdTrainSet = (np.asfarray(knn.metadata[0:sliceSize, 0:-1], float) - newMean[0:-1]) / newSD[0:-1]
            stdTrainSet = np.column_stack([stdTrainSet, knn.metadata[0:sliceSize, -1]])

            stdTestSet = (np.asfarray(testKnn.metadata[:, 0:-1], float) - newMean[0:-1]) / newSD[0:-1]
            stdTestSet = np.column_stack([stdTestSet, testKnn.metadata[:, -1]])

            distanceMatrix = np.sum(np.absolute(stdTestSet[:, None, 0:-1] - stdTrainSet[None, :, 0:-1]), -1)
            ind = np.argpartition(distanceMatrix, k, axis=1)[:, :k]
            predictions = stdTrainSet[ind, -1].astype(int)
            predictions = np.sort(predictions, axis=1)
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(testKnn.labels))
            accuracy = count.item() / testKnn.shape[0]
            print(sliceSize, end=",")
            print(accuracy)
    elif (args == -1):
        testKnn.loadAndInitDataSet(testKnn.inputFile)

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
        accuracy = count.item() / testKnn.shape[0]
        print(k, end=",")
        print(accuracy)

    elif (args == -2):
        validateKnn.loadAndInitDataSet(validateKnn.inputFile)
        accuracyList = []
        distanceMatrix = np.sum(np.absolute(validateKnn.metadata[:, None, 0:-1] != knn.metadata[None, :, 0:-1]), -1)
        for kValue in range(1, knn.k+1):
            #ind = np.argpartition(distanceMatrix, kValue, axis=1)[:, :kValue]
            ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :kValue]
            predictions = knn.metadata[ind, -1]
            result = np.array(mode(predictions, axis=1)[0])
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(validateKnn.labels))
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

        newK = bestK

        testKnn.loadAndInitDataSet(testKnn.inputFile)

        # Combine Train and Validation dataSet
        trainValMetadata = np.concatenate((knn.metadata, validateKnn.metadata))

        distanceMatrix = np.sum(np.absolute(testKnn.metadata[:, None, 0:-1] != trainValMetadata[None, :, 0:-1]), -1)
        ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :newK]
        predictions = trainValMetadata[ind, -1]
        predictions = np.sort(predictions, axis=1)

        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(testKnn.labels))
        res = count.item() / testKnn.shape[0]
        print(res)
    elif (args == -3):
        testKnn.loadAndInitDataSet(testKnn.inputFile)

        batchSize = knn.shape[0] / 10
        for i in range(1, 11):
            sliceSize = int(batchSize * i)
            distanceMatrix = np.sum(testKnn.metadata[:, None, 0:-1] != knn.metadata[None, 0:sliceSize, 0:-1], -1)
            ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
            predictions = knn.metadata[0:sliceSize, :][ind, -1]
            predictions = np.sort(predictions, axis=1)
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(testKnn.labels))
            accuracy = count.item() / testKnn.shape[0]
            print(sliceSize, end=",")
            print(accuracy)
    else:
        testKnn.loadAndInitDataSet(testKnn.inputFile)

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
            for i in range(testKnn.shape[0]):
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