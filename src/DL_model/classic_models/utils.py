#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/7 7:14 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : utils.py
# @desc :

from typing import List

import logging
import os
import random
import re

import torch
import numpy as np
import tqdm


def get_embedding_matrix_and_vocab(w2v_file, skip_first_line=True, include_special_tokens=True):
    """
    Construct embedding matrix

    Args:
        embed_dic : word-embedding dictionary
        skip_first_line : 是否跳过第一行
    Returns:
        embedding_matrix: return embedding matrix (numpy)
        embedding_matrix: return embedding matrix
    """
    embedding_dim = None

    # 先遍历一次，得到一个vocab list, 和向量的list
    vocab_list = []
    vector_list = []
    with open(w2v_file, "r", encoding="utf-8") as f_in:
        for i, line in tqdm.tqdm(enumerate(f_in)):
            if skip_first_line:
                if i == 0:
                    continue

            line = line.strip()
            if not line:
                continue

            line = line.split(" ")
            w_ = line[0]
            vec_ = line[1: ]
            vec_ = [float(w.strip()) for w in vec_]
            # print(w_, vec_)

            if embedding_dim == None:
                embedding_dim = len(vec_)
            else:
                # print(embedding_dim, len(vec_))
                assert embedding_dim == len(vec_)

            vocab_list.append(w_)
            vector_list.append(vec_)

    # 添加两个特殊的字符： PAD 和 UNK
    if include_special_tokens:
        vocab_list = ["[PAD]", "[UNK]"] + vocab_list
        # 随机初始化两个向量,
        pad_vec_ = (np.random.rand(embedding_dim).astype(np.float32) * 0.05).tolist()
        unk_vec_ = (np.random.rand(embedding_dim).astype(np.float32) * 0.05).tolist()
        vector_list = [pad_vec_, unk_vec_] + vector_list

    return vocab_list, vector_list

def tokenizer(text):
    """
    decs 分词，我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
         注意这种分词会会去除一些字符如（ -
    :param text str: xxxxx
    :return:
    """
    regEx = re.compile('[\\W]+')#我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')#[\u4e00-\u9fa5]中文范围
    sentences = regEx.split(text)
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

def build_vocab(lines)->List:
    vocab_list = set()
    for line in lines:
        tokens = tokenizer(line)
        vocab_list.update(tokens)
    return ["[PAD]", "[UNK]"] +list(vocab_list)

def get_vocab(vocab_file, skip_first_line=False):
    vocab_list = []
    with open(vocab_file, "r", encoding="utf-8") as f_in:
        for i, line in tqdm.tqdm(enumerate(f_in)):
            if skip_first_line:
                if i == 0:
                    continue

            line = line.strip()
            if not line:
                continue

            word = line.split(" ")[0]
            vocab_list.append(word)

    return vocab_list


def  init_logger(log_file=None, log_file_level=logging.NOTSET):
    """
    :param log_file: str, 文件目录
    :param log_file_level: str, 日志等级
    """
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file,encoding="utf-8")
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger

def seed_everything(seed):
    """
    :param seed: ## 随机种子
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_labels(label_file):
    return [label.strip() for label in open(label_file, 'r', encoding='utf-8')]




