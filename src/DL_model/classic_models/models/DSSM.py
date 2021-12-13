#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/7 3:30 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : DSSM.py
# @desc :
import torch
import torch.nn as nn
import numpy as np

class DSSM(nn.Module):
    def __init__(self, args):
        self.args = args
        super(DSSM, self).__init__()

        ## embedding 权重的初始化
        if args.random_init_w2v:
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim)
        else:
            w2v_matrix = np.asarray(args.vector_list)
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim).from_pretrained(torch.FloatTensor(w2v_matrix), freeze=False)

        self.fc1 = nn.Linear(args.embed_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=args.dropout_rate)

        self.loss_fct = nn.BCEWithLogitsLoss()
        self._initialize_weights()


    def forward(self, q1, q2, labels=None):
        a = self.word_embedding(q1).sum(1)
        b = self.word_embedding(q2).sum(1)

        a = self.act(self.fc1(a))
        # a = self.dropout(a)
        a = self.act(self.fc2(a))
        # a = self.dropout(a)
        a = self.act(self.fc3(a))
        # a = self.dropout(a)

        b = self.act(self.fc1(b))
        # b = self.dropout(b)
        b = self.act(self.fc2(b))
        # b = self.dropout(b)
        b = self.act(self.fc3(b))
        # b = self.dropout(b)

        cosine = torch.cosine_similarity(a, b, dim=1, eps=1e-8)
        logits = self.act(cosine)
        logits = torch.clamp(logits,0,1)

        pred = logits.detach() # [batch_size]
        pred[pred<0.5]=0
        pred[pred>=0.5]=1

        output = (logits, pred)
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())
            output += (loss,)

        return output   #(logits, pred, loss)

    def _initialize_weights(self):
        """
        desc:线性层初始化
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)



