#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.1:$CUDA_HOME


model_path="/search/odin/liruihong/TextEncoder/model_output"
python eval_model.py \
    --model_path="" \
    --annotate_file="" \
    --eval_res_file=""