#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/20 11:37 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : SiaGRU.py
# @desc :

import torch
import torch.nn as nn
import numpy as np

class SiaGRU(nn.Module):
    def __init__(self, args):
        super(SiaGRU, self).__init__()
        self.args = args

        # 初始化embedding
        if args.random_init_w2v:
            self.word_embedding = nn.Embedding(
                len(args.vocab_list),
                args.embed_dim,
            )
        else:
            self.w2v_matrix = np.asarray(args.vector_list)
            self.word_embedding = nn.Embedding(
                len(args.vocab_list),
                args.embed_dim,
            ).from_pretrained(torch.FloatTensor(self.w2v_matrix), freeze=False)

        ## -------------encoder layer---------------
        if args.rnn_type.lower() == "lstm":
            self.encoder = nn.LSTM(args.embed_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True,
                                   bidirectional=args.bidirectional, bias=False)
        elif args.rnn_type.lower() == "gru":
            self.encoder = nn.GRU(args.embed_dim, args.hidden_size, num_layers=args.num_layers, batch_first=True,
                                  bidirectional=args.bidirectional, bias=False)
        else:
            raise Exception("rnn_type is not in lstm or gru")

        ## ------------dropout---------------
        self.dropout = nn.Dropout(p=args.dropout_rate)

        ## -------------线性层-------------
        self.classifier = nn.Linear(args.max_seq_len, args.num_class)

        ## -----------loss function------------
        self.loss_fct = nn.CrossEntropyLoss()

        # ------------init weight-------------
        self.apply(self._init_weight)

    def _init_weight(self, m):
        for name, param in self.encoder.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, q1:torch.Tensor, q2:torch.Tensor, labels=None):
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        p_encode = self.word_embedding(q1)
        h_endoce = self.word_embedding(q2)
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)

        encoding1, _ = self.encoder(p_encode)
        encoding2, _ = self.encoder(h_endoce)

        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))  # [bs,seq_len,1]
        logits = self.classifier(sim.squeeze(dim=-1))  # [bs, num_classes]
        probabilities = nn.functional.softmax(logits, dim=-1)
        pred = probabilities.argmax(dim=-1)  # [batch_size]
        output = (logits, pred)

        ## loss compute
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output += (loss,)
        return output