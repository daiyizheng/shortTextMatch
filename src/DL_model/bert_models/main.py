#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/14 12:35 上午
# @Author : daiyizheng
# @Version：V 0.1
# @File : main.py
# @desc :
import sys, os
sys.path.insert(0, "./")
import argparse


import torch

from src.DL_model.bert_models.utils.common import init_logger, seed_everything
from src.DL_model.bert_models.data_loader import load_and_cache_examples
from src.DL_model.bert_models.configs import MODEL_CLASSES
from src.DL_model.bert_models.trainer import Trainer

def main(args):
    init_logger()
    seed_everything(args.seed)
    args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    ## tensorboardx
    # args.tensorboardx_path = os.path.join(args.tensorboardx_path, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(args.tensorboardx_path):
        os.makedirs(args.tensorboardx_path)

    ## 模型输出目录
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    tokenizer = MODEL_CLASSES[args.model_name][2].from_pretrained(
        args.model_name_or_path
    )

    tmp_res = tokenizer.tokenize("[MASK]")
    print("tmp_res: ", tmp_res)
    ##--------------------------------- 加载数据 -----------------------------------------
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    print("train_dataset: ", len(train_dataset))
    print("dev_dataset: ", len(dev_dataset))
    print("test_dataset: ", len(test_dataset))

    ##------------------- 加载训练 ------------------------------
    trainer = Trainer(
        args, train_dataset, dev_dataset, test_dataset
    )

    if args.do_train:
        trainer.train()
    #
    if args.do_eval:
        trainer.load_model()
        trainer.evaluate()

    if args.do_predict:
        trainer.load_model()
        trainer.predict()



if __name__ == '__main__':
    argv_len = len(sys.argv)
    params = sys.argv
    model_name = None
    for index, param in enumerate(params):
        if param=="--model_name":
            if index+1>=argv_len:
                raise ValueError("params is error")
            if params[index+1] not in ['bert', 'albert', 'roberta', 'distilbert', 'xlnet', 'electra', 'nezha']:
                raise ValueError('use model name is not in (bert, albert, roberta, distilbert, xlnet, electra, nezha)')
            else:
                model_name = params[index+1]
    if model_name is None:
        raise ValueError("model_name parameter can not be empty")

    parser = argparse.ArgumentParser()
    # -------------------------  Data Arguments ---------------------------------------
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_name", default=None, required=True, type=str, help="load models name")
    parser.add_argument("--model_name_or_path", default=None, required=True, type=str, help="The path of pretrain model")
    parser.add_argument("--train_path", default=None, required=True, type=str, help="The train input data dir")
    parser.add_argument("--dev_path", default=None, required=True, type=str, help="The dev input data dir")
    parser.add_argument("--test_path", default=None, required=False, type=str, help="The test input data dir")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load models")
    parser.add_argument('--tensorboardx_path', type=str, default="./output/logs", help="tensorboardx logging path")

    parser.add_argument("--random_init_w2v", action="store_true", help="是否直接随机初始化embedding； ")
    parser.add_argument("--w2v_file", default=None, type=str, help="path to pretrained word2vec file")
    parser.add_argument("--vocab_file", default=None, type=str, help="path to pretrained word2vec file")

    ## -------------------------------------- Train Arguments -----------------------------------------
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="dropout_rate ")
    parser.add_argument("--patience", default=20, type=int, help="patience for early stopping ")
    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--num_class", default=2, type=int, help="Classification category")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--metric_key_for_early_stop", default="macro avg__f1-score", type=str, help="metric name for early stopping ")


    ### ---------------------------------------- 选择参数 -------------------------------
    parser.add_argument("--loss_function", default="CrossEntropyLoss", choices=["CrossEntropyLoss", "FocalLoss"],  type=str, help="loss function")
    parser.add_argument("--embeddings_learning_rate", default=1e-5, type=float, help="The learning rate for Adamw.")
    parser.add_argument("--encoder_learning_rate", default=5e-5, type=float, help="The learning rate for Adamw.")
    parser.add_argument("--classifier_learning_rate", default=5e-5, type=float, help="The learning rate for Adamw.")
    parser.add_argument("--lr", default=1e-6, type=float, help="The learning rate for Adam.")

    args = parser.parse_args()
    main(args)