#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
#from .operator import itemgetter
from .letor_metrics import ndcg_score
import logging
import hashlib

"""
分析数据工具
"""
#def cal_auc(preds,labels):
#    """
#    calculate auc
#    Args:
#         preds: list  
#         labels: list
#    """
#    assert len(preds) == len(labels)
#    
#    ziplist = zip(preds,labels)
#    sorted_list = sorted(ziplist,key = itemgetter(0))
#
#    # positive_count, negative_count
#    pos_cnt = 0; neg_cnt = 0
#    cor_pair = 0.0 # correct pair count
#    for pred,label in sorted_list:
#        if float(label) > 0.5:
#            pos_cnt += 1
#            cor_pair += neg_cnt
#        else:
#            neg_cnt += 1
#    return (cor_pair/(pos_cnt*neg_cnt)) if (neg_cnt > 0 and pos_cnt > 0) else 0.0


def cal_NDCG(results,k):
    """
    results: format is:
        [label,qid,featureScore,query,docid]
    """
    columns = ["label","qid","featureScore","sen_id"]
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
    

def load_file(file_name, column_name, skip_head=False):
    """
    载入待评测文件，返回pandas DataTable形式数据
    """
    data = []
    with open(file_name) as fp:
        for line in fp:
            features = []
            firstCol = True
            items = line.strip().split('\t')
            if skip_head ==True:
                skip_head = False 
                continue
            else:
                items = line.strip().split('\t')
                items[2] = int(items[2])
                items[3] = float(items[3])
                data.append(items)

    pd_data = pd.DataFrame(data,columns=column_name)
    return pd_data 
    

def cal_dumpfile_NDCG(file_name, k):
    dataTable = load_file(file_name, column_name=["qid","sentence","label","score"])
    qid_set = set(dataTable.qid)
    totalScore = 0.0
    for qid in qid_set:
        features = dataTable[dataTable['qid'] == qid]['score'].values
        labels = dataTable[dataTable['qid'] == qid]['label'].values
        totalScore += ndcg_score(labels,features,k)
    ndcg_val = totalScore/len(qid_set)
    return ndcg_val


if __name__ == "__main__":
    file_name = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data/kw_title_online_socre"
    ndcg_val = cal_dumpfile_NDCG(file_name, 500)
    logging.info("%s ndcg_val:%f"%(file_name, ndcg_val))


