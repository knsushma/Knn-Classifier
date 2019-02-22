#!/usr/bin/python
import json
import math

from pprint import pprint as pp


class KnnClassifier:
    def __init__(self, k, inputFile, outputFile, stdDataSet):
        self.k = k
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.stdDataSet = stdDataSet

    def getLabelSet(self):
        with open(self.inputFile) as trainingDataFile:
            trainingData = json.load(trainingDataFile)
        return trainingData["metadata"].get("features")[-1][1]

    def standardizeDataSet(self):
        with open(self.inputFile) as trainingDataFile:
            trainingData = json.load(trainingDataFile)
            featureLength = len(trainingData["data"][0])-1
            trainingDataStd = []
            for data in trainingData["data"]:
                features = data[:-1]
                label = data[-1]
                mean = sum(features)/featureLength
                summation = 0.0
                for f in features:
                    summation += pow(f-mean,2)
                sd = math.sqrt(summation/featureLength)
                dataSet = []
                for f in features:
                    dataSet.append((f-mean)/sd)
                dataSet.append(label)
                trainingDataStd.append(dataSet)
            return trainingDataStd

    def findManhattanDistance(self, dataSet1, dataSet2):
        manhattDistance = 0.0
        for i in range(0, 63):
            manhattDistance += abs(dataSet1[i] - dataSet2[i])
        return manhattDistance

    def findKNearestNeighbour(self, testDataSet):
        for testIter,testItem in enumerate(testDataSet[:1]):
            neighbours = []
            for trainingIter,trainingItem in enumerate(self.stdDataSet[:10]):
                manhattDistance = self.findManhattanDistance(trainingItem, testItem)
                neighbours.append([manhattDistance, trainingItem[-1], testItem[-1]])
            neighbours.sort(key=lambda x: x[0])
            kNearest = neighbours[:self.k]
            pp(kNearest)
            lables = self.getLabelSet()



if __name__ == '__main__':
    knnTrainer = KnnClassifier(5, './Resources/digits_train.json', './Resources/digits_output.json', [])
    knnTrainer.stdDataSet = knnTrainer.standardizeDataSet()
    pp(len(knnTrainer.stdDataSet))

    knnTester = KnnClassifier(5, './Resources/digits_test.json', './Resources/digits_output.json', [])
    knnTester.stdDataSet = knnTester.standardizeDataSet()
    pp(len(knnTester.stdDataSet))
    pp(knnTester.stdDataSet[-1])
    knnTrainer.findKNearestNeighbour(knnTester.stdDataSet)
    #knnTrainer.getLabelSet()


