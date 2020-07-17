#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import sklearn
from sklearn import metrics
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")


def eval_model(pred_res_file, ori_file):
    ori_df = pd.read_table(ori_file, encoding="utf-8")
    ori_df['label'] = ori_df['label'].apply(lambda x: 1 if x > -1 else 0)
    pred_df = pd.read_table(pred_res_file, names=["sen_a","sen_b","score"], encoding="utf-8")
    model_auc = metrics.roc_auc_score(ori_df['label'].astype(int).tolist(), pred_df['score'].astype(float).tolist()) 
    logging.info("%s auc:%f"%(pred_res_file, model_auc))


if __name__ == "__main__":
    ori_file = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data_new/merged_all.tsv"
    pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/bert-reduce/fuse-data-epoch10/pred_res"
    #pred_res_file = "/search/odin/liruihong/TextEncoder/model_output/word2vec/pred_res"
    eval_model(pred_res_file, ori_file)