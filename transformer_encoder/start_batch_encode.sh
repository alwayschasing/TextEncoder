#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.1:$CUDA_HOME

transformer_model="/search/odin/liruihong/pretrained_model/pytorch_model/bert-base-chinese"
init_model="/search/odin/liruihong/TextEncoder/model_output/bert-reduce/fuse-data-epoch10"
#encode_data="/search/odin/chyg/query_project/data/query_output"
#encode_data="/search/odin/liruihong/sentence-bert/faiss_project/test.txt"
#output_file="/search/odin/chyg/query_project/data/query_output_embedding"

encode_data=$1
output_file=$2
gpu_id=$3
if [[ -z $gpu_id  ]];then
    gpu_id="4"
fi

python batch_encode.py \
    --gpu_id=$gpu_id \
    --transformer_model=$transformer_model \
    --init_model=$init_model \
    --batch_size=1024 \
    --encode_data=$encode_data \
    --skip_firstline=0 \
    --output_file=$output_file
