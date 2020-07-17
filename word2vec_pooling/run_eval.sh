#!/bin/bash
# word2vec_file="/search/odin/liruihong/tts/multi_attn_model/config_data/100000-small.txt"
word2vec_file="/search/odin/liruihong/word2vec_embedding/tencent_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt"
annotate_file="/search/odin/liruihong/TextEncoder/data_sets/annotate_data/kw_title_score"
eval_res_file="/search/odin/liruihong/TextEncoder/eval_res/word2vec@10"

python EvalWord2Vec.py \
    --word2vec_file=$word2vec_file \
    --annotate_file=$annotate_file \
    --eval_res_file=$eval_res_file
