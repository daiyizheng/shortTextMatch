#!/bin/bash

export task=QYTM
export model_name=SiaGRU
python src/DL_model/${model_name}/main.py \
  --task ${task} \
  --model_name ${model_name} \
  --output_dir ./output/${task}_${model_name}_20211003 \
  --tensorboardx_path  ./output/logs/runs/${task}_${model_name}_20211003  \
  --train_path ./datasets/${task}/train.csv \
  --dev_path ./datasets/${task}/dev.csv \
  --test_path ./datasets/${task}/test.csv \
  --label_file_level_dir ./datasets/${task}/labels_level.txt \
  --label2freq_level_dir ./datasets/${task}/label2freq_level.json \
  --vocab_file ./resources/QYTM_word2vec/epoch_5_window_5_sg_0_hs_1_dim_300_vocab.txt \
  --w2v_file ./resources/QYTM_word2vec/epoch_5_window_5_sg_0_hs_1_dim_300_w2v.vectors \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 50

# nohup bash QYTM_run.sh > ./output/logs/SiaGRU_202110042151.log 2>&1 &
# jobs