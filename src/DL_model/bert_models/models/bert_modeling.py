#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/25 12:23 上午
# @Author : daiyizheng
# @Version：V 0.1
# @File : bert_modeling.py
# @desc :

import torch
import torch.nn as nn
import torch.nn .functional as F

from transformers import BertPreTrainedModel, BertModel, BertConfig

from src.DL_model.bert_models.layers.classifier_layer import Classifier


class ClsBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(ClsBERT, self).__init__(config)
        self.args = args
        self.config = config
        self.args.hidden_size = config.hidden_size
        ##----------------- bert层-----------------
        self.bert = BertModel(config=config)




        #-----------分类层--------------
        self.classifier = Classifier(args,
                                     input_dim=self.args.hidden_size,
                                     num_class=args.num_class)


        ## 损失函数
        self.loss_fct = nn.CrossEntropyLoss()


    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None):
        bert_outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        sequence_output = bert_outputs[0]
        bert_pooled_output = bert_outputs[1]

        logits = self.classifier(bert_pooled_output)
        probabilities = nn.functional.softmax(logits, dim=-1)  # [batch_size, num_class]

        pred = probabilities.argmax(dim=-1)  # [batch_size]
        output = (logits, pred)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.args.num_class), labels.view(-1))
            output += (loss,)
        return output




