from pprint import pprint as pp
import json
import math
import numpy as np
from numpy import *
from collections import Counter
from scipy.stats import mode



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

    k = 30
    # knn = HyperParamKnn(k, "./Resources/digits_train.json")
    # validateKnn = HyperParamKnn(k, "./Resources/digits_val.json")
    # testKnn = HyperParamKnn(k, "./Resources/digits_test.json")

    knn = HyperParamKnn(k, "./Resources/votes_train.json")
    validateKnn = HyperParamKnn(k, "./Resources/votes_val.json")
    testKnn = HyperParamKnn(k, "./Resources/votes_test.json")

    args = 5

    knn.loadAndInitDataSet(knn.inputFile)

    if (args == 1):
        testKnn.loadAndInitDataSet(testKnn.inputFile)
        distanceMatrix = knn.findDistanceMatrix(testKnn)
        ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
        predictions = knn.metadata[ind, -1]
        predictions = np.sort(predictions, axis=1)
        predictedLabels = np.array(mode(predictions, axis=1)[0])
        for index in range(predictions.shape[0]):
            mapper = Counter(predictions[index].tolist())
            for i in knn.labelTypes:
                if (mapper.get(i) == None):
                    print(0, end=",")
                else:
                    print(mapper.get(i), end=",")
            print(predictedLabels[index][0])
        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(testKnn.labels))
        accuracy = count.item() / testKnn.shape[0]
        print(k, end=",")
        print(accuracy)
    elif (args == 2):
        validateKnn.loadAndInitDataSet(validateKnn.inputFile)
        accuracyList = []
        distanceMatrix = knn.findDistanceMatrix(validateKnn)
        for kValue in range(1,knn.k+1):
            ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :kValue]
            predictions = knn.metadata[ind, -1]
            predictions  = np.sort(predictions, axis=1)
            predictedLabels = np.array(mode(predictions, axis=1)[0])
            flatPredictedLabels = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatPredictedLabels == np.array(validateKnn.labels))
            accuracy = count.item() / validateKnn.shape[0]
            print(kValue, end=",")
            print(accuracy)
            accuracyList.append([kValue, accuracy])

        npAccuracy = np.array(accuracyList)
        temp = npAccuracy[npAccuracy[:, 1].argsort()]
        temp = temp[temp[:, 1] == npAccuracy.max(axis=0)[1]]
        bestK = int(temp[0][0])
        print(bestK)

        newK = bestK

        testKnn.loadAndInitDataSet(testKnn.inputFile)
        trainValMetadata = np.concatenate((knn.metadata,validateKnn.metadata))
        knn.metadata = trainValMetadata
        distanceMatrix = knn.findDistanceMatrix(testKnn)
        ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :newK]
        predictions = knn.metadata[ind, -1]
        predictions = np.sort(predictions, axis=1)
        flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
        count = np.sum(flatResult == np.array(testKnn.labels))
        accuracy = count.item() / testKnn.shape[0]
        print(accuracy)
    elif(args == 3):
        testKnn.loadAndInitDataSet(testKnn.inputFile)
        batchSize = knn.shape[0] / 10
        for i in range(1, 11):
            knn.sliceSize = int(batchSize * i)
            distanceMatrix = knn.findDistanceMatrix(testKnn)
            ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
            predictions = knn.metadata[ind, -1]
            predictions = np.sort(predictions, axis=1)
            flatResult = np.array(mode(predictions, axis=1)[0]).flatten()
            count = np.sum(flatResult == np.array(testKnn.labels))
            accuracy = count.item() / testKnn.shape[0]
            print(knn.sliceSize, end=",")
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

            testLabels = testKnn.metadata[:, -1].copy()
            for index, label in enumerate(testLabels):
                testLabels[index,] = labelMap.get(label)
            testLabels = testLabels.astype(int)

            constant = float(math.pow(10,-5))

            distanceMatrix = knn.findDistanceMatrix(testKnn)
            kDistanceMatrix = np.sort(distanceMatrix, axis=1)[:, :k]
            ind = np.argsort(distanceMatrix, axis=1, kind='stablesort')[:, :k]
            predictedLabels = knn.metadata[ind, -1]
            predictedLabels = (predictedLabels==category[0])*1

            weightedDistance = (1 / ((np.square(kDistanceMatrix)) + constant))
            weightedSum = np.sum(weightedDistance, axis=1)

            confidence = np.sum(weightedDistance * predictedLabels,axis=1)/weightedSum
            rocMatrix = np.column_stack((testLabels,confidence))
            rocMatrixSorted = rocMatrix[np.argsort([-rocMatrix[:, 1]])][0]
            posNegCount = Counter(rocMatrixSorted[:,0].astype(int))
            posCount = posNegCount.get(1)
            negCount = posNegCount.get(0)

            TP = 0
            FP = 0
            last_TP = 0
            for i in range(testKnn.shape[0]):
                if (i > 1) and (rocMatrixSorted[i,1] != rocMatrixSorted[i-1,1]) and ( testLabels[i] == 0) and ( TP > last_TP):
                    FPR = FP / negCount
                    TPR = TP / posCount
                    print(FPR, end=",")
                    print(TPR)
                    last_TP = TP
                if testLabels[i] == 1:
                    TP += 1
                else:
                    FP += 1
            FPR = FP / negCount
            TPR = TP / posCount
            print(FPR, end=",")
            print(TPR)