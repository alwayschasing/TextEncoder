"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
import torch
import math
import logging
from datetime import datetime
import sys
import os
import argparse
import pickle
from sentence_transformers import SentenceTransformer


def load_texts(input_file):
    texts = []
    with open(input_file, "r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip().split('\t')[0]
            texts.append(text)
    return texts


def extract_embeddings(encoder_model, input_file, output_file):
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    batch_size = 64
    model = SentenceTransformer(encoder_model)
    logging.info("input_file:%s"%(input_file))
    logging.info("Start Read input data")
    input_texts = load_texts(input_file)
    logging.info("Finish Read input data")
    text_embeddings = model.encode(sentences=input_texts, batch_size=batch_size)
    wfp = open(output_file,"wb")
    pickle.dump(text_embeddings, wfp)
    wfp.close()
    logging.info("Finish embedding extraction")
    

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    input_file = "/search/odin/liruihong/video_data/pengpai_data"
    output_file = "/search/odin/liruihong/video_data/pengpai_embeddings"
    encoder_model = "/search/odin/liruihong/TextEncoder/model_output/bert-base/fuse-data-epoch5-utf8"
    extract_embeddings(encoder_model, input_file, output_file)

