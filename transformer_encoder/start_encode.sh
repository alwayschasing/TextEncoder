#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.1:$CUDA_HOME

transformer_model="/search/odin/liruihong/pretrained_model/pytorch_model/bert-base-chinese"
init_model="/search/odin/liruihong/TextEncoder/model_output/bert-reduce/fuse-data-epoch10"
encode_data="/search/odin/liruihong/article_data/titles_test.tsv"
output_file="/search/odin/liruihong/article_data/titles_embeddings_test"

python run_encode.py \
    --gpu_id="3" \
    --transformer_model=$transformer_model \
    --init_model=$init_model \
    --batch_size=32 \
    --encode_data=$encode_data \
    --skip_firstline=0 \
    --output_file=$output_file
