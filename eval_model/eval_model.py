#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import sklearn
from sklearn import metrics
import pandas as pd
import logging
from analysis import ndcg_score
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")


def eval_model(pred_res_file, ori_file):
    ori_df = pd.read_table(ori_file, encoding="utf-8")
    ori_df['label'] = ori_df['label'].apply(lambda x: 1 if x > 0 else 0)
    pred_df = pd.read_table(pred_res_file, names=["sen_a","sen_b","score"], encoding="utf-8")
    model_auc = metrics.roc_auc_score(ori_df['label'].astype(int).tolist(), pred_df['score'].astype(float).tolist())
    logging.info("%s auc:%f"%(pred_res_file, model_auc))


def load_data(pred_res_file, annotate_file):
    data = []
    with open(annotate_file, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            items = line.split('\t') 
            sen_a,sen_b,label = items[0:3]
            data.append([sen_a, sen_b, int(label)])
    with open(pred_res_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        for idx, line in enumerate(lines):
            items = line.strip().split('\t')
            score = float(items[2])
            data[idx].append(score)
    return data


def eval_model_by_ndcg(pred_res_file, annotate_file, k=10):
    data = load_data(pred_res_file, annotate_file)
    column_name = ["qid", "sentence", "label", "score"]
    datatable = pd.DataFrame(data, columns=column_name)

    qid_set = set(datatable.qid)
    totalScore = 0.0
    for qid in qid_set:
        features = datatable[datatable['qid'] == qid]['score'].values
        labels = datatable[datatable['qid'] == qid]['label'].values
        totalScore += ndcg_score(labels, features, k)
    ndcg_val = totalScore/len(qid_set)
    logging.info("%s ndcg:%s"%(pred_res_file, ndcg_val))

if __name__ == "__main__":
    ori_file = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data_new/merged_all.tsv"
    #ori_file = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data/kw_titlecontent_label"
    #pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/bert-reduce/fuse-data-epoch10/pred_res"
    #pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/bert-base/fuse-data-epoch10-utf8/pred_res"
    pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/word2vec/pred_res"
    #pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/bert-reduce/albert-fuse-data-epoch10/pred_res"
    #pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/bert-reduce/labeled-fuse-data-epoch10/pred_res_old"
    #pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/bert-reduce/fuse-data-epoch10/pred_res_old"
    eval_model(pred_res_file, ori_file)
    #ori_file = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data/kw_titlecontent_label"
    #pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/bert-reduce/fuse-data-epoch10/pred_res"
    #eval_model_by_ndcg(pred_res_file, ori_file, k = 10)
