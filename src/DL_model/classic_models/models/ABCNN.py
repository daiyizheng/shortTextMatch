#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/20 11:35 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : ABCNN.py
# @desc :

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ABCNN(nn.Module):
    def __init__(self, args):
        super(ABCNN, self).__init__()
        self.args = args

        ## embedding 权重的初始化
        if args.random_init_w2v:
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim)
        else:
            w2v_matrix = np.asarray(args.vector_list)
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim).from_pretrained(
                torch.FloatTensor(w2v_matrix), freeze=False)

        self.linear_size = args.linear_size
        self.num_layer = args.num_layer
        self.conv = nn.ModuleList(
            [Wide_Conv(args.max_seq_len, args.embed_dim, args.device) for _ in range(self.num_layer)])

        self.fc = nn.Sequential(
            nn.Linear(args.embed_dim * (1 + self.num_layer) * 2, self.linear_size),
            nn.LayerNorm(self.linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_size, args.num_class),
        )

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, q1, q2, labels=None):
        mask1, mask2 = q1.eq(0), q2.eq(0)  # [batch_size, seq_len]
        res = [[], []]
        q1_encode = self.word_embedding(q1)  # [batch_size, seq_len, embed_dim]
        q2_encode = self.word_embedding(q2)  # [batch_size, seq_len, embed_dim]
        # eg: s1 => res[0]
        # (batch_size, seq_len, dim) => (batch_size, dim)
        # if num_layer == 0
        res[0].append(F.avg_pool1d(q1_encode.transpose(1, 2), kernel_size=q1_encode.size(1)).squeeze(-1))  # all-ap
        res[1].append(F.avg_pool1d(q2_encode.transpose(1, 2), kernel_size=q2_encode.size(1)).squeeze(-1))
        for i, conv in enumerate(self.conv):
            o1, o2 = conv(q1_encode, q2_encode, mask1, mask2)  # [batch_size, seq_len, embed_dim]
            res[0].append(F.avg_pool1d(o1.transpose(1, 2), kernel_size=o1.size(1)).squeeze(-1))  # [batch_size, dim]
            res[1].append(F.avg_pool1d(o2.transpose(1, 2), kernel_size=o2.size(1)).squeeze(-1))  # [batch_size, dim]
            o1, o2 = attention_avg_pooling(o1, o2, mask1, mask2)  # #[batch_size, seq_len, embed_dim]
            q1_encode, q2_encode = o1 + q1_encode, o2 + q2_encode
        # batch_size * (dim*(1+num_layer)*2) => batch_size * linear_size
        x = torch.cat([torch.cat(res[0], 1), torch.cat(res[1], 1)], 1)  # [batch_size,4*embed_dim]
        logits = self.fc(x)  # [batch_size, num_class]

        probabilities = nn.functional.softmax(logits, dim=-1)  # [batch_size, num_class]

        pred = probabilities.argmax(dim=-1)  # [batch_size]
        output = (logits, pred)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output += (loss,)
        return output  # (logits, pred, loss)

class Wide_Conv(nn.Module):
    def __init__(self, seq_len, embeds_size, device="gpu"):
        super(Wide_Conv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        self.W = nn.Parameter(torch.randn((seq_len, embeds_size)))
        nn.init.xavier_normal_(self.W)
        self.W.to(device)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=[1, 1], stride=1)
        self.tanh = nn.Tanh()

    def forward(self, sent1, sent2, mask1, mask2):
        '''
        sent1, sent2: batch_size * seq_len * dim
        '''
        # sent1, sent2 = sent1.transpose(0, 1), sent2.transpose(0, 1)
        # => A: batch_size * seq_len * seq_len
        A = match_score(sent1, sent2, mask1, mask2)#[batch_size, seq_len, seq_len]
        # attn_feature_map1: batch_size * seq_len * dim
        attn_feature_map1 = A.matmul(self.W)# A：[batch_size, seq_len, seq_len] * W:[seq_len, embed_dim]--》 [batch_size ,seq_len ,embed_dim]
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W) # A.transpose(1, 2) [batch_size, seq_len, seq_len] --》 [batch_size ,seq_len ,embed_dim]
        # x1: batch_size * 2 *seq_len * dim
        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1) # sent1.unsqueeze(1) [batch_size, 1, seq_len, embed_dim]  attn_feature_map1.unsqueeze(1) [batch_size, 1, seq_len, embed_dim] ->  [batch_size, 2, seq_len, embed_dim]
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1) #self.conv(x1)  [batch_size, 2, seq_len, embed_dim]  -> [batch_size, 1, seq_len, embed_dim]
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2

def match_score(s1, s2, mask1, mask2):
    '''
    s1, s2:  batch_size * seq_len  * dim  计算匹配分数
    '''
    batch, seq_len, dim = s1.shape
    s1 = s1 * mask1.eq(0).unsqueeze(2).float() # [batch_size, seq_len, dim]
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()
    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len, 1) #[batch_size, seq_len, seq_len, embed_dim]
    s2 = s2.unsqueeze(1).repeat(1, seq_len, 1, 1) #[batch_size, seq_len, seq_len, embed_dim]
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)## 最后一个维度求2范式[batch_size, seq_len, seq_len]
    return 1.0 / (1.0 + a)

def attention_avg_pooling(sent1, sent2, mask1, mask2):
    # A: batch_size * seq_len * seq_len
    A = match_score(sent1, sent2, mask1, mask2) # [batch_size, seq_len, seq_len]
    weight1 = torch.sum(A, -1) #[batch_size, seq_len]
    weight2 = torch.sum(A.transpose(1, 2), -1)
    s1 = sent1 * weight1.unsqueeze(2)
    s2 = sent2 * weight2.unsqueeze(2)#[batch_size,seq_len,embed_dim]
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1) #[batch_size,embed_dim,seq_len] w-ap
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2