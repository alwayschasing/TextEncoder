# -*- coding: cp936 -*-

import pandas as pd

from letor_metrics import dcg_score
from letor_metrics import ndcg_score


def loadFile(filePath, q=None):
    featuresID, data, y, qid = [], [], [], []

    firstRow = True
    file = open(filePath)
    for line in file:
        # apply condition
        if q != None and not ('qid:{}'.format(q) in line): continue

        # process line
        features = []
        firstCol = True
        for word in line.split():
            if firstCol:
                y.append(int(word))
                firstCol = False
            elif word.startswith('qid'):
                qid.append(int(word.split(':')[1]))
            elif word.startswith('#'):
                break
            else:
                if firstRow:
                    fID = word.split(':')[0]
                    featuresID.append(fID)
                    features.append(float(word.split(':')[1]))
                else:
                    features.append(float(word.split(':')[1]))
        firstRow = False
        data.append(features)
    featuresTable = pd.DataFrame(data, columns=featuresID)
    featuresTable['qid'] = qid
    featuresTable['Label'] = y
    return featuresTable


def DCGMetricCal(dataTable, k):
    """
    计算根据单特征排序后DCG值
    :param dataTable:
    :param k:
    :return:
    """
    Metric = "DCG@" + str(k)
    qidSet = set(dataTable.qid)
    featureSet = list(dataTable.columns)
    featureSet.remove('qid')
    featureSet.remove('Label')

    featureScores = []
    for featureID in featureSet:
        totalScore = 0
        for qid in qidSet:
            feature = dataTable[dataTable['qid'] == qid][featureID].values
            label = dataTable[dataTable['qid'] == qid]['Label'].values
            totalScore += dcg_score(label, feature, k)
        featureScore = totalScore / len(qidSet)
        featureScores.append(featureScore)
        print [Metric, featureID, featureScore]

    return featureScores


def NDCGMetricCal(dataTable, k):
    """
    计算根据单特征排序后NDCG值
    :param dataTable:
    :param k:
    :return:
    """
    Metric = "NDCG@" + str(k)
    qidSet = set(dataTable.qid)
    featureSet = list(dataTable.columns)
    featureSet.remove('qid')
    featureSet.remove('Label')

    featureScores = []
    for featureID in featureSet:
        totalScore = 0
        for qid in qidSet:
            feature = dataTable[dataTable['qid'] == qid][featureID].values
            label = dataTable[dataTable['qid'] == qid]['Label'].values
            totalScore += ndcg_score(label, feature, k)
        featureScore = totalScore / len(qidSet)
        featureScores.append(featureScore)
        print [Metric, featureID, featureScore]

    return featureScores

def StaFeatureAnalysis(DataPath, ResultFeilePath):
    """
    计算根据单特征统计信息
    :param dataTable:
    :param k:
    :return:
    """
    dataTable = loadFile(DataPath)
    FeatureCorr = dataTable.describe().T
    FeatureCorr.to_csv(ResultFeilePath)


def CorrFeatureAnalysis(DataPath, ResultFeilePath):
    """
    计算特征之间相关性
    :param dataTable:
    :param k:
    :return:
    """
    dataTable = loadFile(DataPath)
    FeatureCorr = dataTable.corr()
    FeatureCorr.to_csv(ResultFeilePath)


def DisFeatureAnalysis(DataPath, ResultFeilePath):
    """
    特征在不同Label上的平均值和标准差，计算区分度
    :param dataTable:
    :param k:
    :return:
    """

    dataTable = loadFile(DataPath)
    featureSet = list(dataTable.columns)
    featureSet.remove('qid')
    featureSet.remove('Label')

    dis2to1, dis2to0, dis1to0 = [], [], []
    FeatureAvg = dataTable.groupby('Label').mean()
    FeatureStd = dataTable.groupby('Label').std()
    for feature in featureSet:
        StdDis2to1 = abs(FeatureStd.at[2, feature] - FeatureStd.at[1, feature])
        if StdDis2to1 != 0:
            dis2to1.append(abs(FeatureAvg.at[2, feature] - FeatureAvg.at[1, feature]) /StdDis2to1)
        else:
            dis2to1.append(0)

        StdDis2to0 = abs(FeatureStd.at[2, feature] - FeatureStd.at[0, feature])
        if StdDis2to0 != 0:
            dis2to0.append(abs(FeatureAvg.at[2, feature] - FeatureAvg.at[0, feature]) /StdDis2to0)
        else:
            dis2to0.append(0)

        StdDis1to0 = abs(FeatureStd.at[1, feature] - FeatureStd.at[0, feature])
        if StdDis1to0 != 0:
            dis1to0.append(abs(FeatureAvg.at[1, feature] - FeatureAvg.at[0, feature]) /StdDis1to0)
        else:
            dis1to0.append(0)

    FinalScore = pd.DataFrame()
    FinalScore['dis2to1'] = dis2to1
    FinalScore['dis2to0'] = dis2to0
    FinalScore['dis1to0'] = dis1to0
    FinalScore.to_csv(ResultFeilePath)
    pass


def RankFeatureAnalysis(DataPath, ResultFeilePath):
    """
    根据排序准则对特征进行分析
    :param dataTable:
    :param k:
    :return:
    """
    dataTable = loadFile(DataPath)
    FinalScore = pd.DataFrame()
    featureScore = DCGMetricCal(dataTable, 1)
    FinalScore['NDCG@1'] = featureScore
    featureScore = NDCGMetricCal(dataTable, 3)
    FinalScore['NDCG@3'] = featureScore
    featureScore = NDCGMetricCal(dataTable, 5)
    FinalScore['NDCG@5'] = featureScore
    featureScore = NDCGMetricCal(dataTable, 10)
    FinalScore['NDCG@10'] = featureScore
    FinalScore.to_csv(ResultFeilePath)


FilePath = r'D:\Work\FeatureAnalysis\AnalysisResult\train.txt'
ResultPath = r"D:\Work\FeatureAnalysis\AnalysisResult\DisFeatureAnalysis.csv"
DisFeatureAnalysis(FilePath, ResultPath)

