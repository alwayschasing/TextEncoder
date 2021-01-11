#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.1:$CUDA_HOME

transformer_model="/search/odin/liruihong/pretrained_model/pytorch_model/bert-base-chinese"
init_model="/search/odin/liruihong/TextEncoder/model_output/bert-reduce/fuse-data-epoch10"
input_dir="/search/odin/liruihong/onlinedoc_encoding/data"
output_dir="/search/odin/liruihong/onlinedoc_encoding/embedding_data"
python_bin="/search/odin/environment/anaconda3/bin/python"

for text_name in title content
do
    timestr=`date -d '-1 day' +"%Y%m%d"`
    input_file="${input_dir}/article_${timestr}"
    output_file="${output_dir}/${text_name}_embedding_${timestr}"
    echo "process $text_name, input:${input_file}, output:${output_file}"
    $python_bin gen_kvfmt_embedding.py \
        --gpu_id="7" \
        --transformer_model=$transformer_model \
        --init_model=$init_model \
        --batch_size=32 \
        --input_file=$input_file\
        --text_name=$text_name \
        --skip_firstline=0 \
        --output_file=$output_file
done

