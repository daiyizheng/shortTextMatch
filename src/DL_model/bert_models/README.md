
## BERT
```shell script
python src/DL_model/bert_models/main.py \
--task LCQMC \
--model_name bert \
--model_name_or_path resources/bert/bert-base-chinese \
--model_dir output/QYTM_bert_20211003 \
--tensorboardx_path output/logs/runs/QYTM_bert_20211003 \
--train_path datasets/LCQMC/LCQMC_train.csv \
--dev_path datasets/LCQMC/LCQMC_dev.csv \
--test_path datasets/LCQMC/LCQMC_test.csv \
--train_batch_size 64 \
--eval_batch_size 64 \
--max_seq_len 103 \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs 50 \
--save_steps 50 \
--logging_steps 50 \
--lr 1e-7
```

## NEZHA 只能单GPU跑模型
```shell script
CUDA_VISIBLE_DEVICES=1 python src/DL_model/bert_models/main.py \
--task LCQMC \
--model_name nezha \
--model_name_or_path resources/nezha/NEZHA-Base \
--model_dir output/QYTM_NEZHA_20211003 \
--tensorboardx_path output/logs/runs/QYTM_NEZHA_20211003 \
--train_path datasets/LCQMC/LCQMC_train.csv \
--dev_path datasets/LCQMC/LCQMC_dev.csv \
--test_path datasets/LCQMC/LCQMC_test.csv \
--train_batch_size 64 \
--eval_batch_size 64 \
--max_seq_len 103 \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs 50 \
--save_steps 50 \
--logging_steps 50 \
--lr 1e-7
```

## RoBerta
```shell script
python src/DL_model/bert_models/main.py \
--task LCQMC \
--model_name roberta \
--model_name_or_path resources/roberta/chinese-roberta-wwm-ext \
--model_dir output/QYTM_NEZHA_20211003 \
--tensorboardx_path output/logs/runs/QYTM_NEZHA_20211003 \
--train_path datasets/LCQMC/LCQMC_train.csv \
--dev_path datasets/LCQMC/LCQMC_dev.csv \
--test_path datasets/LCQMC/LCQMC_test.csv \
--train_batch_size 64 \
--eval_batch_size 64 \
--max_seq_len 103 \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs 50 \
--save_steps 50 \
--logging_steps 50 \
--lr 1e-7
```

## albert
```shell script
python src/DL_model/bert_models/main.py \
--task LCQMC \
--model_name albert \
--model_name_or_path resources/albert/albert-base-v2 \
--model_dir output/QYTM_NEZHA_20211003 \
--tensorboardx_path output/logs/runs/QYTM_NEZHA_20211003 \
--train_path datasets/LCQMC/LCQMC_train.csv \
--dev_path datasets/LCQMC/LCQMC_dev.csv \
--test_path datasets/LCQMC/LCQMC_test.csv \
--train_batch_size 64 \
--eval_batch_size 64 \
--max_seq_len 103 \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs 50 \
--save_steps 50 \
--logging_steps 50 \
--lr 1e-7
```

## distilbert
```shell script
python src/DL_model/bert_models/main.py \
--task LCQMC \
--model_name distilbert \
--model_name_or_path resources/distilbert/distilbert-base-uncased \
--model_dir output/QYTM_NEZHA_20211003 \
--tensorboardx_path output/logs/runs/QYTM_NEZHA_20211003 \
--train_path datasets/LCQMC/LCQMC_train.csv \
--dev_path datasets/LCQMC/LCQMC_dev.csv \
--test_path datasets/LCQMC/LCQMC_test.csv \
--train_batch_size 64 \
--eval_batch_size 64 \
--max_seq_len 103 \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs 50 \
--save_steps 50 \
--logging_steps 50 \
--lr 5e-5
```