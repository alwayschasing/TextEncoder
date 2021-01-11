#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
# from analysis import analysis
import csv
import scipy
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import logging
from datetime import datetime
import sys
import os
import argparse
import logging
import hashlib
sys.path.append("..")
from analysis import analysis
import argparse


logging.basicConfig(level=logging.DEBUG, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

def load_annotate_data(annotate_file, delimiter='\t'):
    annotate_dataset = []
    with open(annotate_file,"r") as fp:
        for line in fp:
            items = line.strip().split(delimiter)
            annotate_dataset.append(items)
    return annotate_dataset

def load_title_content_data(annotate_file, delimiter='\t'):
    annotate_dataset = []
    with open(annotate_file,"r", encoding="utf-8") as fp:
        for line in fp:
            line = line.encode("utf-8").decode("unicode_escape")
            items = line.rstrip('\n').split(delimiter)
            if len(items) != 5:
                print(line)
            kw, title, content, label, docid = items
            annotate_dataset.append([kw, title + ' ' + content, label])
    return annotate_dataset



def eval_model(annotate_file, model_name, eval_res_file):
    # annotate_dataset = load_annotate_data(annotate_file)
    annotate_dataset = load_title_content_data(annotate_file)
    results = []

    idindex = {}
    corpus = []
    count = 0
    for query, sen, label in annotate_dataset:
        id_gen = hashlib.md5()
        id_gen.update(query.encode('utf-8'))
        query_id = id_gen.hexdigest()
        if query_id not in idindex:
            corpus.append(query)
            idindex[query_id] = count
            count += 1

        id_gen = hashlib.md5()
        id_gen.update(sen.encode('utf-8'))
        sen_id = id_gen.hexdigest() 
        if sen_id not in idindex:
            corpus.append(sen)
            idindex[sen_id] = count
            count += 1
    
    model = SentenceTransformer(model_name)
    #word_embedding_model = models.Transformer(model_name)
    #pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
    #                           pooling_mode_mean_tokens=True,
    #                           pooling_mode_cls_token=False,
    #                           pooling_mode_max_tokens=False)
    #model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    corpus_embeddings = model.encode(corpus)

    for query, sen, label in annotate_dataset:
        label = int(label)

        id_gen = hashlib.md5()
        id_gen.update(query.encode('utf-8'))
        query_id = id_gen.hexdigest()

        id_gen = hashlib.md5()
        id_gen.update(sen.encode('utf-8'))
        sen_id = id_gen.hexdigest() 

        logging.debug('query:%s'%(query))
        logging.debug('idindex[query_id]:%d'%(idindex[query_id]))
        logging.debug('sen:%s'%(sen))
        logging.debug('idindex[sen_id]:%d'%(idindex[sen_id]))
        query_vec = corpus_embeddings[idindex[query_id]]
        sen_vec = corpus_embeddings[idindex[sen_id]]
        sim_score = scipy.spatial.distance.cdist([query_vec],[sen_vec], "cosine")[0] 
        results.append((label,query_id, sim_score, sen_id))

    fp = open(eval_res_file,"w", encoding="utf-8")
    writer = csv.writer(fp)
    ndcg = analysis.cal_NDCG(results,10)
    writer.writerow([model_path, ndcg])
    fp.close()

def model_predict(input_file, output_file):
    pass


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description="Text Similarity")
    parser.add_argument('--annotate_file', action='store', type=str, required=True, help="annotate_file")
    parser.add_argument('--model_path', required=True, type=str, help="model_path")
    parser.add_argument('--eval_res_file', default="eval_res_file", type=str, help="eval_res_file")
    args = parser.parse_args()
    annotate_file = args.annotate_file
    model_path = args.model_path
    eval_res_file = args.eval_res_file
    eval_model(annotate_file, model_path, eval_res_file)
    
