#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/9/8 9:34 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : bert_mode_pretraing.py
# @desc :
## 预训练模型

import sys
sys.path.append("./")

import pandas as pd
import numpy as np

import argparse
import logging
import math
import os
import random

import datasets
from datasets import load_dataset
from tqdm.auto import tqdm
from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import (CONFIG_MAPPING, MODEL_MAPPING, AdamW, AutoConfig, AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, SchedulerType, get_scheduler, set_seed)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# from pprint import pprint
# pprint(MODEL_TYPES, width=3, compact=True)


# class TrainConfig:
#     train_file= 'mlm_data.csv'
#     validation_file = 'mlm_data.csv'
#     validation_split_percentage= 5
#     pad_to_max_length= True
#     model_name_or_path= 'roberta-base'
#     config_name= 'roberta-base'
#     tokenizer_name= 'roberta-base'
#     use_slow_tokenizer= True
#     per_device_train_batch_size= 2
#     per_device_eval_batch_size= 2
#     learning_rate= 5e-5
#     weight_decay= 0.0
#     num_train_epochs= 5 # change to 5
#     max_train_steps= None
#     gradient_accumulation_steps= 1
#     lr_scheduler_type= 'constant_with_warmup'
#     num_warmup_steps= 0
#     output_dir= 'output'
#     seed= 2021
#     model_type= 'roberta'
#     max_seq_length= None
#     line_by_line= False
#     preprocessing_num_workers= 1
#     overwrite_cache= True
#     mlm_probability= 0.15
#
# config = TrainConfig()




def main(args):
    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)

    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    elif args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        perplexity = math.exp(torch.mean(losses))

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
if __name__ == "__main__":
    # df_train = pd.read_csv("../../../datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/fold_0/train.txt", delimiter="\t", header=None)
    # df_dev = pd.read_csv("../../../datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/fold_0/dev.txt", delimiter="\t", header=None)
    # df_test = pd.read_csv("../../../datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/fold_0/test.txt", delimiter="\t", header=None)
    # df_train = df_train.append(df_dev)
    # df_train.reset_index(drop=True)
    # df_train.columns=['id','text','label']
    # df_test.columns=['id','text']
    # df_train[['text']].to_csv("../../../datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/pretrain_data/mlm_data.csv", index=None, encoding="utf_8_sig")
    # df_test[['text']].to_csv("../../../datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/pretrain_data/mlm_data_val.csv", index=None, encoding="utf_8_sig")
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name_or_path", default="resources/bert/chinese-roberta-wwm-ext/", type=str,
                       required=True, help="bert path")
    parse.add_argument("--train_file",
                       default="datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/pretrain_data/mlm_data.csv",
                       type=str, required=True, help="train path")
    parse.add_argument("--validation_file",
                       default="datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/pretrain_data/mlm_data_val.csv",
                       type=str, required=True, help="dev path")
    parse.add_argument("--validation_split_percentage", default=5, type=int, help="")
    parse.add_argument("--pad_to_max_length", default=True, type=bool, help="pad_to_max_length")
    parse.add_argument("--config_name", default="hfl/chinese-roberta-wwm-ext", type=str, help="config name")
    parse.add_argument("--tokenizer_name", default="hfl/chinese-roberta-wwm-ext", type=str, help="tokenizer name")
    parse.add_argument("--use_slow_tokenizer", default=True, type=bool, help="use_slow_tokenizer")
    parse.add_argument("--per_device_train_batch_size", default=2, type=int, help="per_device_train_batch_size")
    parse.add_argument("--per_device_eval_batch_size", default=2, type=int, help="per_device_train_batch_size")
    parse.add_argument("--learning_rate", default=5e-5, type=float, help="learning_rate")
    parse.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay")
    parse.add_argument("--num_train_epochs", default=5, type=int, help="num_train_epochs")
    parse.add_argument("--max_train_steps", default=None, type=int, help="max_train_steps")
    parse.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient_accumulation_steps")
    parse.add_argument("--lr_scheduler_type", default="constant_with_warmup", type=str, help="lr_scheduler_type")
    parse.add_argument("--num_warmup_steps", default=0, type=int, help="new bert path")
    parse.add_argument("--output_dir", default="resources/bert/chinese-roberta-wwm-ext-pretrain/", type=str,
                       help="output_dir")
    parse.add_argument("--seed", default=2021, type=str, help="seed")
    parse.add_argument("--model_type", default="roberta", type=str, help="model_type")
    parse.add_argument("--max_seq_length", default=None, type=int, help="max_seq_length")
    parse.add_argument("--line_by_line", default=False, type=bool, help="nline_by_line")
    parse.add_argument("--preprocessing_num_workers", default=1, type=int, help="preprocessing_num_workers")
    parse.add_argument("--overwrite_cache", default=True, type=bool, help="overwrite_cache")
    parse.add_argument("--mlm_probability", default=0.15, type=float, help="mlm_probability")

    args = parse.parse_args()
    main(args)
    """
    CUDA_VISIBLE_DEVICES=1 python src/bert_models/model_process/bert_mode_pretraing.py \
    --model_name_or_path resources/bert/chinese-roberta-wwm-ext/ \
    --train_file datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/pretrain_data/mlm_data.csv \
    --validation_file datasets/phase_1/bert_mapping_splits/chinese-roberta-wwm-ext/counts/pretrain_data/mlm_data_val.csv \
    --output_dir resources/bert/chinese-roberta-wwm-ext-pretrain/ \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --max_seq_length 512 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4
    """