#!/usr/bin/env python
# -*- encoding:utf-8 -*-
from word2vec_encoder import Word2VecEncoder


def load_sentence_pair(input_file, skip_head=True):
    sentence_pairs = []
    with open(input_file, "r", encoding="utf-8") as fp:
        lines = fp.readlines() 
        for idx, line in enumerate(lines):
            if skip_head == True and idx == 0:
                continue
            items = line.strip().split('\t')
            sentence_pairs.append([items[0], items[1]])
    return sentence_pairs

def main():
    input_file = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data_new/merged_all.tsv"
    output_file = "/search/odin/liruihong/TextEncoder/model_output/word2vec/pred_res"
    word2vec_file = "/search/odin/liruihong/word2vec_embedding/tencent_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt"
    stopwords_file = None
    sentence_pairs = load_sentence_pair(input_file)
    model = Word2VecEncoder(word2vec_file=word2vec_file, stopwords_file=stopwords_file)
    pred_res = model.predict_cosine_similarity(sentence_pairs)
    with open(output_file, "w", encoding="utf-8") as fp:
        for idx, item in enumerate(pred_res):
            pair = sentence_pairs[idx]
            sim = item
            fp.write("%s\t%s\t%f\n"%(pair[0], pair[1], sim))


if __name__ == "__main__":
    main()
    
