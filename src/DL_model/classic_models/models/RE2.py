#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/20 11:36 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : RE2.py
# @desc :

import torch.nn as nn
import torch
import numpy as np

from src.DL_model.classic_models.models.layers import ModuleList, ModuleDict
from src.DL_model.classic_models.models.layers.encoder import Encoder
from src.DL_model.classic_models.models.layers.pooling import Pooling
from src.DL_model.classic_models.models.layers.alignment import registry as alignment
from src.DL_model.classic_models.models.layers.fusion import registry as fusion
from src.DL_model.classic_models.models.layers.connection import registry as connection
from src.DL_model.classic_models.models.layers.prediction import registry as prediction

class RE2(nn.Module):
    def __init__(self, args):
        super(RE2, self).__init__()
        self.args = args

        # -------- 初始化embedding ----------
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

        ## ---------------------Encoder layer、Alignment layer、Fusion layer--------------
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(args, args.embed_dim if i == 0 else args.embed_dim + args.hidden_size),
            'alignment': alignment[args.alignment](
                args, args.embed_dim + args.hidden_size if i == 0 else args.embed_dim + args.hidden_size * 2),
            'fusion': fusion[args.fusion](
                args, args.embed_dim + args.hidden_size if i == 0 else args.embed_dim + args.hidden_size * 2),
        }) for i in range(args.blocks)])

        ## -----------------Residual connection layer ------
        self.connection = connection[args.connection]()
        ##-------------------- pool layer -----------------
        self.pooling = Pooling()

        ## ---------------- classifier layer ----------
        self.classifier = prediction[args.prediction](args)

        ## --------损失函数
        self.loss_fct = nn.CrossEntropyLoss()


    def forward(self, q1:torch.Tensor, q2:torch.Tensor, labels=None):
        mask_q1 = torch.ne(q1, 0).unsqueeze(2).to(self.args.device)
        mask_q2 = torch.ne(q2, 0).unsqueeze(2).to(self.args.device)
        q1 = self.word_embedding(q1)
        q2 = self.word_embedding(q2)
        res_q1, res_q2 = q1, q2

        for i, block in enumerate(self.blocks):
            if i > 0:
                q1 = self.connection(q1, res_q1, i)
                q2 = self.connection(q2, res_q2, i)
                res_q1, res_q2 = q1, q2
            q1_enc = block['encoder'](q1, mask_q1)
            q2_enc = block['encoder'](q2, mask_q2)
            q1 = torch.cat([q1, q1_enc], dim=-1)
            q2 = torch.cat([q2, q2_enc], dim=-1)
            align_q1, align_q2 = block['alignment'](q1, q2, mask_q1, mask_q2)
            q1 = block['fusion'](q1, align_q1)
            q2 = block['fusion'](q2, align_q2)

        q1 = self.pooling(q1, mask_q1)
        q2 = self.pooling(q2, mask_q2)
        logits = self.classifier(q1, q2)
        probabilities = nn.functional.softmax(logits, dim=-1)
        pred = probabilities.argmax(dim=-1)  # [batch_size]
        output = (logits, pred)

        ## loss compute
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output += (loss,)
        return output