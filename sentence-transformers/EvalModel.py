#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
# from analysis import analysis
import csv
import scipy
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import sys
import os
import argparse
import logging
import hashlib
sys.path.append("..")
from analysis import analysis

"""
because bert is memory limit,
run model.test just generate the sentence vector,
here use the sentence cal ndcg on ltr test data
"""

logging.basicConfig(level=logging.INFO, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

def load_annotate_data(annotate_file, delimiter='\t'):
    annotate_dataset = []
    with open(annotate_file,"r") as fp:
        for line in fp:
            items = line.strip().split(delimiter)
            annotate_dataset.append(items)
    return annotate_dataset


def eval_model(annotate_file, model_name, eval_res_file):
    annotate_dataset = load_annotate_data(annotate_file)
    results = []

    id_gen = hashlib.md5()
    idindex = {}
    corpus = []
    count = 0
    for query, sen, label in annotate_dataset:
        id_gen.update(query[0:128].encode('utf-8'))
        query_id = id_gen.hexdigest()
        if query_id not in idindex:
            corpus.append(query)
            idindex[query_id] = count
            count += 1

        id_gen.update(sen[0:128].encode('utf-8'))
        sen_id = id_gen.hexdigest() 
        if sen_id not in idindex:
            corpus.append(sen)
            idindex[sen_id] = count
            count += 1
    
    embedder = SentenceTransformer(model_name)
    corpus_embeddings = embedder.encode(corpus)

    for query, sen, label in annotate_dataset:
        label = int(label)

        id_gen.update(query[0:128].encode('utf-8'))
        query_id = id_gen.hexdigest()
        id_gen.update(sen[0:128].encode('utf-8'))
        sen_id = id_gen.hexdigest() 

        query_vec = corpus_embeddings[idindex[query_id]]
        sen_vec = corpus_embeddings[idindex[sen_id]]
        sim_score = scipy.spatial.distance.cdist([query],[sen], "cosine")[0] 
        results.append((label,query_id, sim_score, sen_id))

    fp = open(eval_res_file,"w", encoding="utf-8")
    writer = csv.writer(fp)
    ndcg = analysis.cal_NDCG(results,10)
    writer.writerow([model_path, ndcg])
    fp.close()


if __name__ == "__main__":
    annotate_file = ""
    model_path = ""
    eval_res_file = ""
    eval_model(annotate_file, model_path, eval_res_file)
    



        


    

