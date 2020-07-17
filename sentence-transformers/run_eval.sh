#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.1:$CUDA_HOME


#model_path="/search/odin/zhuguangnan/SBERT/trans_data/bert_base_chinese/"
#model_path="/search/odin/liruihong/TextEncoder/model_output/bert-base/multi-content-title-epoch15"
#annotate_file="/search/odin/liruihong/TextEncoder/data_sets/annotate_data/kw_content_label"
#eval_res_file="/search/odin/liruihong/TextEncoder/eval_res/content-title@15"

python EvalModel.py \
    --model_path=$model_path \
    --annotate_file=$annotate_file \
    --eval_res_file=$eval_res_file
