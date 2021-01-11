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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import math
import logging
from datetime import datetime
import sys
import os
import numpy as np
import argparse
from encoder_model import BertEncoder


def load_sentences(input_file, skip_firstline=False):
    sentences = []
    with open(input_file) as fp:
        lines = fp.readlines()
        for idx, line in enumerate(lines):
            if skip_firstline == True and idx == 0:
                continue
            sen = line.strip().split('\t')[0]
            sentences.append(sen)
    return sentences


def main(args):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[])

    transformer_model = args.transformer_model
    # Read the dataset
    batch_size = args.batch_size
    #model_save_path = os.path.join(model_output_dir, "bert-base", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    word_embedding_size = 768
    reduce_output_size = 128
    model = BertEncoder(transformer_model, word_embedding_size, reduce_output_size)
    if args.init_model is not None:
        model.load(args.init_model)
    else:
        raise ValueError("init_model is None")

    logging.info("Read encode dataset")
    encode_data_file = args.encode_data
    sentences = load_sentences(encode_data_file)
    sen_embeddings = model.encode(sentences=sentences, batch_size=args.batch_size)
    output_file = args.output_file
    sen_embeddings = np.asarray(sen_embeddings)
    np.savetxt(output_file, sen_embeddings, delimiter='\t')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Similarity")
    parser.add_argument('--gpu_id', default='0', type=str, help="gpu_id num")
    parser.add_argument('--transformer_model', action='store', default="", type=str, required=True, help="model name")
    parser.add_argument('--init_model', type=str, default=None, help="init model dir")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--encode_data', type=str, help="encode_data path")
    parser.add_argument('--skip_firstline', type=int, default=0, help="whether skip first line of data")
    parser.add_argument('--output_file', type=str, help='output file')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args)

