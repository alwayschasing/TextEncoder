#!/usr/bin/env python
# -*- encoding:utf-8 -*-
from analysis import analysis
import logging

file_name = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data/kw_title_online_socre"
ndcg_val = analysis.cal_dumpfile_NDCG(file_name, 10)
print("%s ndcg_val:%f"%(file_name, ndcg_val))

def load_pred_res():
    pass

def load_annotate_file():
    pass

def load_data(pred_res_file, annotate_file):
    pass

def eval_model_predict():
    data = load_data()
    column_name = ["qid", "sentence", "label", "score"]
    datatable = pd.DataFrame(data, columns=column_name)

    qid_set = set(datatable.qid)
    totalScore = 0.0
    for qid in qid_set:
        features = datatable[datatable['qid'] == qid]['score'].values
        labels = datatable[datatable['qid'] == qid]['label'].values
        totalScore += ndcg_score(labels, features, k)
    ndcg_val = totalScore/len(qid_set)
