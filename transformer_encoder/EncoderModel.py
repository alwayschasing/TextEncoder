import torch
import logging
from torch import nn
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Optional
import os
import numpy as np
from .Pooling import Pooling
from .util import import_from_string, batch_to_device, http_get
from .Evaluator import Evaluator

class DimReduceModel(nn.Sequential):
    def __init__(self, input_size, output_size):
        self.linear1 = nn.Linear(in_features=input_size, out_features=output_size, bias=False)
        self.act1 = nn.Tanh()
        super(DimReduceModel, self).__init__(self.linear1, act1)

class BertEncoder(nn.Module):

    def __init__(self, bert_path: str, word_embedding_size: int, reduce_output_size:int, device: str = None, max_seq_length: int = 256):
        super(BertEncoder, self).__init__()
        self.do_lower_case = do_lower_case
        if max_seq_length > 510:
            logging.warning("BERT only allows a max_seq_length of 510 (512 with special tokens). Value will be set to 510")
            max_seq_length = 510
        self.max_seq_length = max_seq_length


        self.bert = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.pooling_model = Pooling(word_embedding_dimension=word_embedding_size,
                                     pooling_mode_cls_token=False,
                                     pooling_mode_max_token=False,
                                     pooling_mode_mean_token=True)
        #self.dimreduce_model = nn.Linear(in_features=word_embedding_size, out_features=reduce_output_size, bias=True)
        self.dimreduce_model = DimReduceModel(input_size=word_embedding_size, output_size=reduce_output_size)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Use pytorch device: {}".format(device))
        self.device = torch.device(device)
        self.to(device)

    def forward(self, features):
        output_states = self.bert(**features)
        output_tokens = output_states[0]
        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': output_tokens, 'cls_token_embeddings': cls_tokens, 'attention_mask': features['attention_mask']})
        if len(output_states) > 2:
            features.update({'all_layer_embeddings': output_states[2]})
        features = self.pooling_model(features)
        reduce_output = self.dimreduce_model(features["sentence_embedding"])
        features.update({"sentence_embedding": reduce_output})
        return features

    def smart_batching_collate(self, batch):
        num_texts = len(batch[0][0])
        labels = []
        paired_texts = [[] for _ in range(num_texts)]
        max_seq_len = [0] * num_texts
        for tokens, label in batch:
            labels.append(label)
            for i in range(num_texts):
                paired_texts[i].append(tokens[i])
                max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

        features = []
        for idx in range(num_texts):
            max_len = max_seq_len[idx]
            feature_lists = {}

            for text in paired_texts[idx]:
                sentence_features = self.get_sentence_features(text, max_len)

                for feature_name in sentence_features:
                    if feature_name not in feature_lists:
                        feature_lists[feature_name] = []

                    feature_lists[feature_name].append(sentence_features[feature_name])

            for feature_name in feature_lists:
                #feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))
                feature_lists[feature_name] = torch.cat(feature_lists[feature_name])

            features.append(feature_lists)
        return {'features': features, 'labels': torch.stack(labels)}

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenizes a text and maps tokens to token-ids
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def get_sentence_features(self, tokens: List[int], pad_seq_length: int):
        """
        Convert tokenized sentence in its embedding ids, segment ids and mask

        :param tokens:
            a tokenized sentence
        :param pad_seq_length:
            the maximal length of the sequence. Cannot be greater than self.sentence_transformer_config.max_seq_length
        :return: embedding ids, segment ids and mask for the sentence
        """
        pad_seq_length = min(pad_seq_length, self.max_seq_length) + 2  ##Add Space for CLS + SEP token
        return self.tokenizer.prepare_for_model(tokens, max_length=pad_seq_length, pad_to_max_length=True, return_tensors='pt')

    def get_word_embedding_dimension(self) -> int:
        return self.bert.config.hidden_size

    def save(self, output_path: str):
        torch.save(self.state_dict(), output_path)

    def load(self, input_path: str):
        self.load_state_dict(torch.load(input_path))

    def evaluate(self, evaluator: Evaluator, output_path: str = None):
        """
        Evaluate the model

        :param evaluator:
            the evaluator
        :param output_path:
            the evaluator can write the results to this path
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        return evaluator(self, output_path)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if score > self.best_score and save_best_model:
                self.save(output_path)
                self.best_score = score

    def fit(self,
            dataloader,
            loss_model,
            evaluator: Evaluator,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler_name: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params : Dict[str, object ]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            local_rank: int = -1
            ):

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            if os.listdir(output_path):
                raise ValueError("Output directory ({}) already exists and is not empty.".format(
                    output_path))

        dataloader.collate_fn = self.smart_batching_collate()
        device = self.device
        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = len(dataloader)

        num_train_steps = int(steps_per_epoch * epochs)
        param_optimizer = list(loss_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
        if local_rank != -1:
            t_total = t_total // torch.distributed.get_world_size()


        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler_name, warmup_steps=warmup_steps, t_total=t_total)

        global_step = 0
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0
            loss_model.zero_grad()
            loss_model.train()

            for batch_data in trange(dataloader, desc="Iteration", smoothing=0.05):
                features, labels = batch_to_device(batch_data, self.device)
                loss_value = loss_model(features, labels)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                    loss_model.zero_grad()
                    loss_model.train()

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)

    def _get_scheduler(self, optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def encode(self, sentences: List[str], batch_size: int = 8, show_progress_bar: bool = None, output_value: str = 'sentence_embedding', convert_to_numpy: bool = True) -> List[ndarray]:
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel()==logging.INFO or logging.getLogger().getEffectiveLevel()==logging.DEBUG)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Batches")

        for batch_idx in iterator:
            batch_tokens = []

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, len(sentences))

            longest_seq = 0

            for idx in length_sorted_idx[batch_start: batch_end]:
                sentence = sentences[idx]
                tokens = self.tokenize(sentence)
                longest_seq = max(longest_seq, len(tokens))
                batch_tokens.append(tokens)

            features = {}
            for text in batch_tokens:
                sentence_features = self.get_sentence_features(text, longest_seq)

                for feature_name in sentence_features:
                    if feature_name not in features:
                        features[feature_name] = []
                    features[feature_name].append(sentence_features[feature_name])

            for feature_name in features:
                #features[feature_name] = torch.tensor(np.asarray(features[feature_name])).to(self.device)
                features[feature_name] = torch.cat(features[feature_name]).to(self.device)

            with torch.no_grad():
                out_features = self.forward(features)
                embeddings = out_features[output_value]

                if output_value == 'token_embeddings':
                    #Set token embeddings to 0 for padding tokens
                    input_mask = out_features['input_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                if convert_to_numpy:
                    embeddings = embeddings.to('cpu').numpy()

                all_embeddings.extend(embeddings)

        reverting_order = np.argsort(length_sorted_idx)
        all_embeddings = [all_embeddings[idx] for idx in reverting_order]
        return all_embeddings


