"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
import scipy
import torch
import math
import threading
import logging
from datetime import datetime
import sys
import os
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pickle
import zmq
import zmq.decorators as zmqd
from zmq.utils import jsonapi
from LoggerUtil import set_logger
import faiss
import numpy as np


def load_texts(input_file):
    texts = []
    with open(input_file, "r") as fp:
        for line in fp:
            text = line.strip().split('\t')[0]
            texts.append(text)
    return texts

def extract_text_embeddings(input_texts_file, output_file):
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.StreamHandler()])

    batch_size = args.batch_size

    model = SentenceTransformer(args.init_model)
    logging.info("input_data:%s"%(args.input_data))
    input_file = args.input_data
    logging.info("Read input data")
    input_texts = load_texts(input_file)
    text_embeddings = model.encode(sentences=input_texts, batch_size=batch_size)

class SenTransEncoder(object):
    def __init__(self, init_model):
        self.model = SentenceTransformer(init_model)

    def encode(self, sentence, pooling_strategy="avg_pooling"):
        text_embeddings = self.model.encode(sentences=[sentence])
        return text_embeddings[0]

class SenTransServer(threading.Thread):
    def __init__(self, sentrans_encoder, index_vecs_file, index_data_file, listen_port=5021, logger=logging.StreamHandler()):
        super(SenTransServer, self).__init__()
        self.encoder = sentrans_encoder
        self.logger = logger
        self.port = listen_port
        self.index_vecs = np.asarray(self.load_index_bin(index_vecs_file), dtype=np.float32)
        self.index_data = self.load_index_data(index_data_file)
        self.logger.info("index_bin size:%d, index_data size:%d"%(len(self.index_vecs), len(self.index_data)))
        assert len(self.index_vecs) == len(self.index_data)
        self.vec_size = 768
        self.build_index(self.index_vecs)
        self.logger.info("Finish SenTransServerServer Init.")

    def build_index(self, index_data):
        "normalize data"
        self.logger.info("Start build index")
        normalize_data = normalize(index_data)
        self.index = faiss.IndexFlatIP(self.vec_size)
        self.index.add(normalize_data)
        self.logger.info("Finish build index")

    def load_index_bin(self, index_bin_file):
        fp = open(index_bin_file, "rb")
        data = pickle.load(fp)
        fp.close()
        return data

    def load_index_data(self, index_data_file):
        index_data = []
        with open(index_data_file, "r") as fp:
            for line in fp:
                items = line.strip().split('\t')
                texts = items[0]
                docid = items[1]
                index_data.append((texts, docid))
        return index_data

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.REP)
    def _run(self,_,backend_sock):
        backend_sock.bind('tcp://127.0.0.1:%d' % self.port)
        self.logger.info("bind word2vec_server socket:%s:%d"%("tcp://127.0.0.1",self.port))
        while True:
            """server job""" 
            try:
                message = backend_sock.recv_json()
            except Exception:
                self.logger.error("recv exception")
            else:
                text = message["query"]
                self.logger.info("[recv text] %s"%(text))
                text_embedding = self.encoder.encode(text)
                text_embedding = normalize(np.asarray([text_embedding], dtype=np.float32))
                dis, idx = self.index.search(text_embedding, 10)
                res = []
                for i,d in enumerate(dis[0]):
                    neighbor_text, docid = self.index_data[idx[0][i]]
                    self.logger.debug("%s\t%s\t%s"%(neighbor_text,str(d),docid))
                    res.append([neighbor_text, str(d), docid])
                backend_sock.send_json({"sentrans_res":res})

    def search(self, query):
        text_embedding = np.asarray([self.encoder.encode(query)], dtype=np.float32)
        text_embedding = normalize(text_embedding)
        print(text_embedding)
        search_res = self.index.search(text_embedding, 10)
        dis,idx = search_res 
        print(search_res)
        res = []
        for i,d in enumerate(dis[0]):
            neighbor_text,docid = self.index_data[idx[0][i]]
            res.append([neighbor_text, d, docid])
        return res

def test():
    handler = logging.StreamHandler()
    logger = set_logger(name="sentrans", verbose=False, handler=handler)
    init_model = "/search/odin/liruihong/server/text_embedding/init_model/fuse-data-epoch10"
    encoder =  SenTransEncoder(init_model) 
    logger.info("build SenTransEncoder")
    listen_port = 5021
    index_vecs_file = "/search/odin/liruihong/article_data/index_data/titles_5d_sentrans.bin"
    index_data_file = "/search/odin/liruihong/article_data/titles_5d.tsv"
    server = SenTransServer(encoder, index_vecs_file, index_data_file, listen_port, logger=logger)
    query = "沉静美学的空间魅力与雅致"
    res = server.search(query) 
    print(res)

def start_server():
    handler = logging.StreamHandler()
    logger = set_logger(name="sentrans", verbose=False, handler=handler)
    init_model = "/search/odin/liruihong/server/text_embedding/init_model/fuse-data-epoch10"
    encoder =  SenTransEncoder(init_model) 
    logger.info("build SenTransEncoder")
    listen_port = 5021
    index_vecs_file = "/search/odin/liruihong/article_data/index_data/titles_5d_sentrans.bin"
    index_data_file = "/search/odin/liruihong/article_data/titles_5d.tsv"
    server = SenTransServer(encoder, index_vecs_file, index_data_file, listen_port, logger=logger)
    #query = "沉静美学的空间魅力与雅致"
    #res = server.search(query) 
    #print(res)
    server.start()
    server.join()


if __name__ == "__main__":
    #test()
    start_server()
