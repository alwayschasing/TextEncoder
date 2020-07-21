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
import argparse
from encoder_model import ALBertEncoder
from losses import CosineSimilarityLoss
from DatasetUtil import SentencesDataset, STSDataReader
from evaluator import EmbeddingSimilarityEvaluator


def main(args):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[])

    transformer_model = args.transformer_model
    # Read the dataset
    batch_size = args.batch_size
    model_output_dir = args.model_output_dir
    #model_save_path = os.path.join(model_output_dir, "bert-base", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    model_save_path = model_output_dir
    word_embedding_size = 768
    reduce_output_size = 128
    model = ALBertEncoder(transformer_model, word_embedding_size, reduce_output_size)
    if args.init_model is not None:
        model.load(args.init_model)

    if args.do_train == 1:
        # Convert the dataset to a DataLoader ready for training
        logging.info("train_data:%s"%(args.train_data))
        logging.info("cache_data:%s"%(args.cached_data))
        train_data_files = args.train_data.split('#')
        cached_data_file = args.cached_data
        logging.info("Read train dataset")
        data_reader = STSDataReader()
        if not os.path.isfile(cached_data_file):
            train_examples = []
            for train_file in train_data_files:
                if os.path.isfile(train_file):
                    logging.info("load train file:%s"%(train_file))
                    now_examples = data_reader.get_examples(train_file)
                    train_examples.extend(now_examples)
            train_data = SentencesDataset(train_examples, model=model)
            torch.save(train_data, args.cached_data)
        else:
            train_data = torch.load(cached_data_file)
            logging.info("Load cached dataset %s"%(cached_data_file))
        logging.info("Build train dataset")
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        # train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)
        train_loss = CosineSimilarityLoss(model=model)


        logging.info("Read dev dataset")
        dev_data_files = args.dev_data.split('#')
        dev_examples = []
        for dev_file in dev_data_files:
            if os.path.isfile(dev_file):
                logging.info("load dev file:%s"%(dev_file))
                now_examples = data_reader.get_examples(dev_file)
                dev_examples.extend(now_examples)
        dev_data = SentencesDataset(examples=dev_examples, model=model)
        logging.info("Build dev dataset")
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
        evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

        # Configure the training
        num_epochs = args.num_epochs
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        summary_dir = args.model_output_dir
        summary_writer = SummaryWriter(summary_dir)
        logging.info("Start training")
        # Train the model
        model.fit(dataloader=train_dataloader,
                  loss_model=train_loss,
                  evaluator=evaluator,
                  epochs=num_epochs,
                  evaluation_steps=args.evaluation_steps,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path,
                  summary_writer=summary_writer)

    if args.do_predict == 1:
        logging.info("Read predict dataset")
        pred_data_file = args.pred_data
        data_reader = STSDataReader()
        input_examples = data_reader.get_examples(pred_data_file, skip_head=args.skip_firstline)
        pred_data = SentencesDataset(input_examples, model=model)
        pred_dataloader = DataLoader(pred_data, shuffle=False, batch_size=batch_size)
        predict_res = model.predict_cosine_similarity(dataloader=pred_dataloader)
        predict_output = os.path.join(args.model_output_dir, "pred_res")
        with open(predict_output, "w", encoding="utf-8") as fp:
            for idx, item in enumerate(predict_res):
                input_example = input_examples[idx]
                fp.write("%s\t%s\t%f\n"%(input_example.texts[0], input_example.texts[1], item))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Similarity")
    parser.add_argument('--gpu_id', default='0', type=str, help="gpu_id num")
    parser.add_argument('--transformer_model', action='store', default="", type=str, required=True, help="model name")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--evaluation_steps', default=1000, type=int)
    parser.add_argument('--model_output_dir', type=str, required=True)
    parser.add_argument('--do_train', default=0, type=int)
    parser.add_argument('--do_eval', default=0, type=int)
    parser.add_argument('--do_predict', default=0, type=int)
    parser.add_argument('--train_data', type=str, help="train_data path")
    parser.add_argument('--cached_data', type=str, help='cached tokenized train data')
    parser.add_argument('--dev_data', type=str, help="dev_data path")
    parser.add_argument('--pred_data', type=str, help="pred_data path")
    parser.add_argument('--init_model', type=str, default=None, help="init model dir")
    parser.add_argument('--skip_firstline', type=int, default=0, help="whether skip first line of data")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args)
