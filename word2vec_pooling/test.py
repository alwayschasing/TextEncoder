#!/usr/bin/env python
# -*- encoding:utf-8 -*-
from gensim.models import KeyedVectors
import jieba

def load_annotate_data(annotate_file, delimiter='\t'):
    annotate_dataset = []
    with open(annotate_file,"r", encoding="utf-8") as fp:
        for line in fp:
            # logging.debug(line.strip())
            items = line.strip().split(delimiter)
            annotate_dataset.append(items)
    return annotate_dataset

word2vec_file="/search/odin/liruihong/tts/multi_attn_model/config_data/test.txt"
model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
vocab = model.vocab
vocab = [v.encode("utf-8").decode("unicode_escape") for v in vocab]
vocab_line = "\t".join(vocab)
print(vocab_line)

annotate_file = "/search/odin/liruihong/TextEncoder/data_sets/annotate_data/kw_title_score"
annotate_dataset = load_annotate_data(annotate_file)
query, sen, label = annotate_dataset[2]

# print("\t".join([query,sen,label]))

words = jieba.lcut(query)
# words = [w.encode("utf-8").decode("unicode_escape") for w in words]
# print(",".join(words))
for w in words:
    if w in model.wv:
        print("[in wv]%s"%(w.encode("utf-8").decode("unicode_escape")))
    else:
        print("[not in wv]%s"%(w.encode("utf-8").decode("unicode_escape")))

words = jieba.lcut(sen)
words = [w.encode("utf-8").decode("unicode_escape") for w in words]
print(",".join(words))
