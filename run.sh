#!/bin/bash
time=$(date "+%Y%m%d%H%M%S")
model_name=("ESIM")
task="LCQMC"
for md in ${model_name[*]}
do
  python src/DL_model/${md}/main.py \
  --task $task \
  --model_name $md \
  --model_dir ./output/"${task}_${md}_${time}" \
  --tensorboardx_path  ./output/logs/runs/"${task}_${md}_${time}"  \
  --train_path ./datasets/LCQMC/LCQMC_train.csv \
  --dev_path ./datasets/LCQMC/LCQMC_dev.csv \
  --test_path ./datasets/LCQMC/LCQMC_test.csv \
  --vocab_file ./resources/word2vec/vocab.txt \
  --w2v_file ./resources/word2vec/token_vec_300.bin \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 50
done