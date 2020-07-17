#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
# from analysis import analysis
import csv
import scipy
import math
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
from word2vec_encoder import Word2VecEncoder
import jieba
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname).1s %(asctime)s] %(message)s", datefmt="%Y-%m-%d_%H:%M:%S")

def load_annotate_data(annotate_file, delimiter='\t'):
    annotate_dataset = []
    with open(annotate_file,"r",encoding="utf-8") as fp:
        for line in fp:
            # logging.debug(line.strip())
            items = line.strip().split(delimiter)
            annotate_dataset.append(items)
    return annotate_dataset


def eval_model(annotate_file, eval_res_file, word2vec_file, stopwords_file=None):
    annotate_dataset = load_annotate_data(annotate_file)
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
    
    #test_str = corpus[10] 
    #logging.info(test_str.encode("utf-8").decode("unicode_escape"))
    ## test_str = test_str.encode("utf-8").decode("unicode_escape")
    #words = jieba.lcut(test_str)
    ## print(words[1])
    ## words = [w.encode("raw_unicode_escape").decode("latin-1") for w in words]
    #words = [w.encode("utf-8").decode("unicode_escape") for w in words]
    #print(words[1])
    model = Word2VecEncoder(word2vec_file=word2vec_file, stopwords_file=stopwords_file)
    corpus_embeddings = model.encode_sentences(corpus)

    for query, sen, label in annotate_dataset:
        label = int(label)

        id_gen = hashlib.md5()
        id_gen.update(query.encode('utf-8'))
        query_id = id_gen.hexdigest()

        id_gen = hashlib.md5()
        id_gen.update(sen.encode('utf-8'))
        sen_id = id_gen.hexdigest() 

        logging.debug('query:%s'%(query.encode("utf-8").decode("unicode_escape")))
        logging.debug('idindex[query_id]:%d'%(idindex[query_id]))
        logging.debug('sen:%s'%(sen.encode("utf-8").decode("unicode_escape")))
        logging.debug('idindex[sen_id]:%d'%(idindex[sen_id]))
        query_vec = corpus_embeddings[idindex[query_id]]
        sen_vec = corpus_embeddings[idindex[sen_id]]
        if query_vec is None: 
            logging.error("%s vec None"%(query.encode("utf-8").decode("unicode_escape")))
            continue
        if sen_vec is None:
            logging.error("%s vec None"%(sen.encode("utf-8").decode("unicode_escape"))) 
            continue
        sim_score = scipy.spatial.distance.cdist([query_vec],[sen_vec], "cosine")[0] 
        results.append((label,query_id, sim_score, sen_id))

    fp = open(eval_res_file,"w", encoding="utf-8")
    writer = csv.writer(fp)
    ndcg = analysis.cal_NDCG(results,10)
    writer.writerow([ndcg])
    fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvalWord2Vec")
    parser.add_argument('--annotate_file', action='store', type=str, required=True, help="annotate_file")
    parser.add_argument('--word2vec_file', required=True, type=str, help="word2vec_file")
    parser.add_argument('--stopwords_file', default=None, type=str, help="stopwords_file")
    parser.add_argument('--eval_res_file', default="eval_res_file", type=str, help="eval_res_file")
    args = parser.parse_args()
    annotate_file = args.annotate_file
    eval_res_file = args.eval_res_file
    eval_model(annotate_file, eval_res_file, args.word2vec_file, args.stopwords_file)
    
