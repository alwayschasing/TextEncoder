#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from operator import itemgetter
from letor_metrics import ndcg_score
import logging

"""
分析数据工具
"""
def loadFile(filePath, q = None):
    """
    载入待评测文件，返回pandas DataTable形式数据
    """
    featuresID, data, y, qid = [], [], [], []
    firstRow = True
    fp = open(filePath)
    for line in fp:
        if q != None and not ( 'qid:{}'.format(q) in line):continue

        features = []
        firstCol = True
        for word in line.split():
            if firstCol:
                y.append(int(word))
                firstCol = False
            elif word.startswith('qid'):
                qid.append(int(word.split(":")[1]))
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
    featuresTable = pd.DataFrame(data,columns=featuresID)
    featuresTable['qid'] = qid 
    featuresTable['Lable'] = y 
    return featuresTable

def cal_auc(preds,labels):
    """
    calculate auc
    Args:
         preds: list  
         labels: list
    """
    assert len(preds) == len(labels)
    
    ziplist = zip(preds,labels)
    sorted_list = sorted(ziplist,key = itemgetter(0))

    # positive_count, negative_count
    pos_cnt = 0; neg_cnt = 0
    cor_pair = 0.0 # correct pair count
    for pred,label in sorted_list:
        if float(label) > 0.5:
            pos_cnt += 1
            cor_pair += neg_cnt
        else:
            neg_cnt += 1
    return (cor_pair/(pos_cnt*neg_cnt)) if (neg_cnt > 0 and pos_cnt > 0) else 0.0

def cal_NDCG_dumpedData(filePath,k,feature_ids):
    """
    计算给定特征id的NDCG@k
    Args:
        filePath: input data file
        k: NDCG@k
        feature_ids: 
    return:
        dict: {"id":score}
    """ 
    result = {}
    dataTable = loadFile(filePath)
    qidSet = set(dataTable.qid)
    featureSet = list(dataTable.columns)
    featureSet.remove('qid')
    featureSet.remove('Lable')

    featureScores = []
    for featureID in feature_ids:
        totalScore = 0.0
        for qid in qidSet:
            features = dataTable[dataTable['qid'] == qid][featureID].values
            labels = dataTable[dataTable['qid'] == qid]['Lable'].values
            totalScore += ndcg_score(labels, features, k)
        featureScore = totalScore/len(qidSet)
        featureScores.append(featureScore)
        result[str(featureID)] = featureScore
    return result

def cal_NDCG(results,k):
    """
    results: format is:
        [label,qid,featureScore,query,docid]
    """
    columns = ["label","qid","featureScore","query","docid"]
    dataTable = pd.DataFrame(results,columns=columns)
    qidSet = set(dataTable.qid)
    logging.info("qidSet size:%d",len(qidSet))
    totalScore = 0.0
    for qid in qidSet:
        features = dataTable[dataTable['qid'] == qid]['featureScore'].values
        labels = dataTable[dataTable['qid'] == qid]['label'].values
        totalScore += ndcg_score(labels,features,k)
    ndcg_val = totalScore/len(qidSet)
    return ndcg_val
    
    
