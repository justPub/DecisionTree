# -*- coding: utf-8 -*-
from math import log
import treePlot
import os
import sys

inputOriginFile = "./data/C4.5.txt"
inputTestFile = "./data/dataTest.txt"

# 创建原始数据集
def createDataSet(filename):
    '''
    :param filename: 训练数据集
    :return: 训练数据集，特征
    '''
    fhandle = open(filename)
    head=fhandle.readline()
    labels=head.split()

    dataSet=[]
    while True:
        lines=fhandle.readline()
        if not lines:
            break
            pass
        dtmp=lines.split()
        dataSet.append(dtmp)
    fhandle.close()
    return dataSet,labels

# 创建测试数据集
def createTestSet(filename):
    '''
    :return: 测试数据集
    '''
    fhandle = open(filename)
    testSet = []
    while True:
        lines = fhandle.readline()
        if not lines:
            break
            pass
        dtmp = lines.split()
        testSet.append(dtmp)
    fhandle.close()
    return testSet

# 测试数据集分类
def classify(inputTree, labels, testTmp):
    '''
    :param inputTree: 决策树
    :param labels: 特征
    :param testTmp: 测试数据
    :return: 分类结果
    '''
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    labelIndex = labels.index(firstStr)
    classResult = {}
    for key in secondDict.keys():
        if testTmp[labelIndex] == key: # 如果特征相同，判断子树的类型，如果是str则停止，dict继续递归遍历
            if type(secondDict[key]).__name__ == 'dict':
                classResult = classify(secondDict[key], labels, testTmp) # 递归遍历
            else:
                classResult = secondDict[key] # 递归出口
    return classResult

# 测试数据集分类
def classifyAll(inputTree, labels, testDataSet):
    '''
    :param inputTree: 决策树
    :param labels: 特征
    :param testDataSet: 测试数据集
    :return: 分类结果
    '''
    classResult = []
    for testDataTmp in testDataSet:
        classResult.append(classify(inputTree, labels, testDataTmp))
    return classResult

# 计算经验熵
def calcEmpiricalEnt(dataSet):
    '''
    :param dataSet: 数据集
    :return: 经验熵
    '''
    dataSetLength = len(dataSet)
    labelCounts = {}
    # 记录特征数
    for i in dataSet:
        currentLabel = i[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    empiricalEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / dataSetLength # 特征出现的概率
        empiricalEnt -= prob * log(prob, 2) # 经验熵
    return empiricalEnt

# 将特征值分离数据集得到子数据集
def splitDataSet(dataSet, axis, value):
    '''
    :param dataSet: 原数据集
    :param axis: 索引--第几个特征值
    :param value: 特征值
    :return: 子数据集
    '''
    retDataSet = []
    for featTmp in dataSet:
        if featTmp[axis] == value:
            reduceFeatTmp = featTmp[:axis] # 去掉当前特征值
            reduceFeatTmp.extend(featTmp[axis+1:]) # 在已有特征列表中加入新列表内容
            retDataSet.append(reduceFeatTmp)
    return retDataSet

# 通过信息增益比选择最优特征
def chooseBestFeatureToSplit(dataSet):
    '''
    :param dataSet: 数据集
    :return: 最优特征
    '''
    numFeatures = len(dataSet[0]) - 1
    originEntropy = calcEmpiricalEnt(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # 特征值
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # 将特征值分离数据集得到子数据集
            prob = len(subDataSet) / float(len(dataSet)) # 计算子数据集的概率
            newEntropy += prob * calcEmpiricalEnt(subDataSet) # 条件熵
            splitInfo += -prob * log(prob, 2) # 数据集关于某个特征值的熵的比
        infoGain = originEntropy - newEntropy # 计算信息增益
        if (splitInfo == 0):
            continue
        infoGainRatio = infoGain / splitInfo # 计算信息增益率
        if (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return bestFeature

# 创建决策树
def createTree(dataSet, labels):
    '''
    :param dataSet: 数据集
    :param labels: 特征
    :return: 决策树
    '''
    classList = [example[-1] for example in dataSet]
    # 只剩一种结果时
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 通过信息增益比选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

if __name__ == '__main__':
    dataSet, labels = createDataSet(inputOriginFile) # 创建原始数据集
    labelsTmp = labels[:]
    desicionTree = createTree(dataSet, labelsTmp) # 创建决策树
    print('desicionTree:\n', desicionTree)
    treePlot.createPlot(desicionTree) # 绘制决策树
    testSet = createTestSet(inputTestFile) # 创建测试数据集
    print('classifyResult:\n', classifyAll(desicionTree, labels, testSet)) # 测试数据集分类