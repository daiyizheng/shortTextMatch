#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/25 12:30 上午
# @Author : daiyizheng
# @Version：V 0.1
# @File : classifier_layer.py
# @desc :

import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, args, input_dim=128, num_class=2):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(args.dropout_rate)
        self.linear = nn.Linear(input_dim, num_class)

    def forward(self, x):
        x = self.dropout(x) # [batch_size, hidden_dim]
        return self.linear(x)
