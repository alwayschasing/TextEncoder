#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import jieba
import numpy as np
from gensim.models import KeyedVectors
import scipy
import logging


class Word2VecEncoder(object):
    def __init__(self, word2vec_file, stopwords_file=None):
        self.model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
        if stopwords_file is not None:
            self.stopwords = load_stopwords(stopwords_file)
        else:
            self.stopwords = set()

    def encode_sentences(self, sentences, pooling_strategy="avg_pooling"):
        sentences_vec = []
        for sen in sentences:
            sen_vec = self.encode(sen)
            sentences_vec.append(sen_vec)
        return sentences_vec

    def encode(self, sentence, pooling_strategy="avg_pooling"):
        words = jieba.lcut(sentence)
        # words = [w.encode("utf-8").decode("unicode_escape") for w in words]
        words = self.filter_stopwords(words)
        words_vec = []
        for w in words:
            if w in self.model.wv:
                words_vec.append(self.model.wv[w])
        words_line = ",".join(words)
        words_line.encode("utf-8").decode("unicode_escape")
        if len(words_vec) == 0:
            words_line = ",".join(words)
            logging.error("[empty words_vec]%s, cut_words:%s"%(sentence.encode("utf-8").decode("unicode_escape"), words_line))
            sen_vec = None
            return sen_vec
        else:
            logging.debug("cut_words:%s"%(words_line))
            
        sen_vec = self.avg_pooling(words_vec)
        sen_vec = sen_vec.tolist()
        return sen_vec

    def avg_pooling(self, words_vec):
        words_vec = np.asarray(words_vec)     
        pooling_vec = np.mean(words_vec, axis=0)
        return pooling_vec
    
    def filter_stopwords(self, words_list):
        words = []
        for w in words_list:
            if w not in self.stopwords:
                words.append(w)
        return words
    
    def predict_cosine_similarity(self, sentence_pair_list):
        pred_res = []
        for sen_a, sen_b in sentence_pair_list:
            embedding_a = self.encode(sen_a) 
            embedding_b = self.encode(sen_b)
            norm_a = np.linalg.norm(embedding_a)
            norm_b = np.linalg.norm(embedding_b)
            cosine_sim = np.dot(embedding_a, embedding_b)/(norm_a * norm_b)
            pred_res.append(cosine_sim)
        return pred_res


def load_stopwords(file_name):
    words = set()
    with open(file_name, encoding="utf-8") as fp:
        for line in fp:
            word = line.strip()
            words.add(word)
    return words


class Word2VecSimilar(object):
    def __init__(self, word2vec_encoder):
        self.encoder = word2vec_encoder

    def cosin_dist(self,sen_a, sen_b):
        vec_a = self.encoder.encode(sen_a)
        vec_b = self.encoder.encode(sen_b)
        dist = scipy.spatial.distance.cosine(vec_a, vec_b) 
        return dist