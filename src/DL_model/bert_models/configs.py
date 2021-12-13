#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/25 12:48 上午
# @Author : daiyizheng
# @Version：V 0.1
# @File : config.py
# @desc :

from src.DL_model.bert_models.models import ClsBERT, ClsRoBERTa, ClsXLNet, ClsELECTRA,ClsDistilBert,ClsAlBert,ClsNezaBert
from transformers import (BertConfig, BertTokenizer,
                          RobertaConfig, RobertaTokenizer,
                          XLNetConfig, XLNetTokenizer,
                          DistilBertConfig, DistilBertTokenizer,
                          ElectraConfig, ElectraTokenizer,
                          AlbertConfig, AlbertTokenizer
                          )


MODEL_CLASSES = {
    "bert":(BertConfig, ClsBERT, BertTokenizer),
    "roberta":(RobertaConfig, ClsRoBERTa, BertTokenizer), ## RobertaTokenizer vocab文件格式变了txt--> json
    "xlnet":(XLNetConfig, ClsXLNet, XLNetTokenizer),
    "electra":(ElectraConfig, ClsELECTRA, ElectraTokenizer),
    "distilbert":(DistilBertConfig, ClsDistilBert, DistilBertTokenizer),
    "albert":(AlbertConfig, ClsAlBert, AlbertTokenizer),
    "nezha":(BertConfig, ClsNezaBert, BertTokenizer)
}

MODEL_PATH_MAP = {
    "bert":"./resources/bert/bert-base-chinese",
    "roberta":"./resources/roberta/chinese-roberta-wwm-ext",
    "xlnet":"./resources/xlnet/chinese-xlnet-base",
    "electra":"./resources/electra/chinese-electra-180g-base-discriminator",
    "distilbert":"./resources/distilbert/distilbert-base-uncased",
    "albert":"./resources/albert/albert_chinese_base",
    "nezha":"./resources/nezha/NEZHA-Base"
}