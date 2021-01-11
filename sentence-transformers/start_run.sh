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
cached_data="/search/odin/liruihong/TextEncoder/cached_data/cached_fuse_data_utf8.train"

#--init_model="/search/odin/liruihong/TextEncoder/model_output/bert-base/2020-05-20_15-56-54" \
#--init_model="/search/odin/liruihong/TextEncoder/model_output/bert-base/fuse-data-epoch5-utf8" \
model_output_dir="/search/odin/liruihong/TextEncoder/model_output/bert-base/fuse-data-epoch10-new"

python TextSimilarity.py \
    --gpu_id="1" \
    --model_name="/search/odin/liruihong/pretrained_model/pytorch_model/bert-base-chinese" \
    --model_output_dir=$model_output_dir \
    --batch_size=32 \
    --num_epochs=10 \
    --train_data=$train_data \
    --cached_data=$cached_data \
    --dev_data=$dev_data \
    --pred_data=$pred_data \
    --do_train=1 \
    --do_predict=0 \
