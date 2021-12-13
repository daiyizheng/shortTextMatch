#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/25 11:23 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : distilbert_modeling.py
# @desc :

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertPreTrainedModel, BertPreTrainedModel

from src.DL_model.bert_models.layers.classifier_layer import Classifier


class ClsDistilBert(DistilBertPreTrainedModel):
    def __init__(self, config, args):
        super(ClsDistilBert, self).__init__(config)
        self.args = args
        self.config = config
        self.args.hidden_size = config.hidden_size
        ##----------------- bert层-----------------
        self.bert = DistilBertModel(config=config)

        # -----------分类层--------------
        self.classifier = Classifier(args,
                                     input_dim=self.args.hidden_size,
                                     num_class=args.num_class)
        ## 损失函数
        self.loss_fct = nn.CrossEntropyLoss()


    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                labels: torch.Tensor = None):
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask)
        shape = input_ids.shape
        # sequence_output = bert_outputs[0]
        hidden_state = bert_outputs[0]
        pooled_output = hidden_state[:, 0]

        logits = self.classifier(pooled_output)
        probabilities = nn.functional.softmax(logits, dim=-1)  # [batch_size, num_class]

        pred = probabilities.argmax(dim=-1)  # [batch_size]
        output = (logits, pred)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.args.num_class), labels.view(-1))
            output += (loss,)
        return output