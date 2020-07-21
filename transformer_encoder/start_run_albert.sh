#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.1:$CUDA_HOME

#train_kw_content="/search/odin/liruihong/TextEncoder/data_sets/kw_content_data/kw_content_300k.train"
#train_title_pair="/search/odin/liruihong/TextEncoder/data_sets/samekw_data/samekw_pairs_train_400k.tsv"
#dev_kw_content="/search/odin/liruihong/TextEncoder/data_sets/kw_content_data/kw_content_30k.dev"
#dev_title_pair="/search/odin/liruihong/TextEncoder/data_sets/samekw_data/samekw_pairs_dev_40k.tsv"
#cache_data="/search/odin/liruihong/TextEncoder/cached_data/cached_sameke_pair_400k.train"
train_data="/search/odin/liruihong/TextEncoder/data_sets/fuse_data/sentence_pair_train.tsv"
dev_data="/search/odin/liruihong/TextEncoder/data_sets/fuse_data/sentence_pair_dev.tsv"
pred_data="/search/odin/liruihong/TextEncoder/data_sets/annotate_data_new/merged_all.tsv"
cached_data="/search/odin/liruihong/TextEncoder/cached_data/transformer_encoder/cached_fuse_data.train"
transformer_model="/search/odin/liruihong/pretrained_model/pytorch_model/albert-base-chinese"

#--init_model="/search/odin/liruihong/TextEncoder/model_output/bert-base/2020-05-20_15-56-54" \
model_output_dir="/search/odin/liruihong/TextEncoder/model_output/bert-reduce/albert-fuse-data-epoch10"

python text_similarity_albert.py \
    --gpu_id="3" \
    --transformer_model=$transformer_model \
    --model_output_dir=$model_output_dir \
    --batch_size=32 \
    --num_epochs=10 \
    --evaluation_steps=1000 \
    --train_data=$train_data \
    --pred_data=$pred_data \
    --skip_firstline=1 \
    --cached_data=$cached_data \
    --dev_data=$dev_data \
    --do_train=1 \
    --do_predict=0
