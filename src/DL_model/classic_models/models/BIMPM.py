#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/20 11:02 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : BIMPM.py
# @desc :

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class BIMPM(nn.Module):
    def __init__(self, args):
        super(BIMPM, self).__init__()
        self.args = args

        ## embedding 权重的初始化
        if args.random_init_w2v:
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim)
        else:
            w2v_matrix = np.asarray(args.vector_list)
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim).from_pretrained(
                torch.FloatTensor(w2v_matrix), freeze=False)

        ## -----encoder layer-----
        rnn_type = None
        if args.rnn_type.lower() == "lstm":
            rnn_type = nn.LSTM
        elif args.rnn_type.lower() == "gru":
            rnn_type = nn.GRU
        else:
            raise Exception("rnn type not in lstm or gru")

        self.context_encoder = rnn_type(input_size=args.embed_dim,
                                       hidden_size=args.hidden_size,
                                       num_layers=args.num_layers,
                                       bidirectional=args.bidirectional,
                                       batch_first=True)

        self.aggregation_encoder = rnn_type(input_size=args.num_perspective * 8,
                                           hidden_size=args.hidden_size,
                                           num_layers=args.num_layers,
                                           bidirectional=True,
                                           batch_first=True)
        # ----- Matching Layer -----
        for i in range(1, 9):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(args.num_perspective, args.hidden_size)))

        ## ------dropout----------
        self.dropout = nn.Dropout(p=args.dropout_rate)

        ## 线性层
        bidirection_num = 1
        if args.bidirectional:
            bidirection_num = 2
        self.args.bidirection_num = bidirection_num

        self.classifier1 = nn.Linear(args.hidden_size * args.num_layers * bidirection_num * 2, args.hidden_size * 2)
        self.classifier2 = nn.Linear(args.hidden_size * 2, args.num_class)

        ## 损失函数
        self.loss_fct = nn.CrossEntropyLoss()

        ## --------- init_weight --------------
        self._init_weight()

    def _init_weight(self):
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.word_embedding.weight.data[0], -0.1, 0.1)
        # ----- Context Representation Layer -----
        nn.init.kaiming_normal_(self.context_encoder.weight_ih_l0)
        nn.init.constant_(self.context_encoder.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_encoder.weight_hh_l0)
        nn.init.constant_(self.context_encoder.bias_hh_l0, val=0)
        nn.init.kaiming_normal_(self.context_encoder.weight_ih_l0_reverse)
        nn.init.constant_(self.context_encoder.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_encoder.weight_hh_l0_reverse)
        nn.init.constant_(self.context_encoder.bias_hh_l0_reverse, val=0)
        # ----- Matching Layer -----
        for i in range(1, 9):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)
        # ----- Aggregation Layer -----
        nn.init.kaiming_normal_(self.aggregation_encoder.weight_ih_l0)
        nn.init.constant_(self.aggregation_encoder.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_encoder.weight_hh_l0)
        nn.init.constant_(self.aggregation_encoder.bias_hh_l0, val=0)
        nn.init.kaiming_normal_(self.aggregation_encoder.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_encoder.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_encoder.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_encoder.bias_hh_l0_reverse, val=0)
        # ----- Prediction Layer ----
        nn.init.uniform_(self.classifier1.weight, -0.005, 0.005)
        nn.init.constant_(self.classifier1.bias, val=0)
        nn.init.uniform_(self.classifier2.weight, -0.005, 0.005)
        nn.init.constant_(self.classifier2.bias, val=0)

    def forward(self, q1, q2, labels=None):
        # ----- Word Representation Layer -----
        # (batch, seq_len) -> (batch, seq_len, word_dim)
        p_encode = self.word_embedding(q1)
        h_endoce = self.word_embedding(q2)

        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)

        # ----- Context Representation Layer -----
        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_encoder(p_encode)
        con_h, _ = self.context_encoder(h_endoce)
        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_p, self.args.hidden_size, dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h, self.args.hidden_size, dim=-1)

        ## -------- Matching compute --------------
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1, self.args.num_perspective)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2, self.args.num_perspective)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1, self.args.num_perspective)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2, self.args.num_perspective)

        # 2. Maxpooling-Matching
        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3, self.args.num_perspective)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4, self.args.num_perspective)
        # (batch, seq_len, l)
        mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        mv_h_max_bw, _ = mv_max_bw.max(dim=1)

        # 3. Attentive-Matching
        # (batch, seq_len1, seq_len2)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)
        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)  # [bs,1, seq_len2, hz]*[bs, seq_len1, seq_len2, 1]->(batch, seq_len1, seq_len2, hidden_size)
        att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)  # [bs, ]
        att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)
        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))  # [bs, seq_len1, hz] , [bs, seq_len1, 1]
        att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))
        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        # (batch, seq_len, l)
        mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw,
                                            self.mp_w5)  # ([bs, seq_len1, hz], [bs, seq_len1, hz])
        mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)

        # 4. Max-Attentive-Matching
        # (batch, seq_len1, hidden_size)
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw, _ = att_p_fw.max(dim=1)
        att_max_p_bw, _ = att_p_bw.max(dim=1)
        # (batch, seq_len, l)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)

        ## --------- Representation information concat -------
        # (batch, seq_len, l * 8)
        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
             mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
             mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)
        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ----- Aggregation Layer -----
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_encoder(mv_p)  # output,(h_n, c_n)-->[bs, seq_len1, bi*h_out], ([n_layers*bi, seq_len1, h_out], )
        _, (agg_h_last, _) = self.aggregation_encoder(mv_h)
        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * self.args.num_layers * self.args.bidirection_num),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.args.hidden_size * self.args.num_layers * self.args.bidirection_num)],
            dim=1)
        x = self.dropout(x)

        # ----- Prediction(classifier) Layer -----
        x = torch.tanh(self.classifier1(x))
        x = self.dropout(x)
        logits = self.classifier2(x)
        probabilities = nn.functional.softmax(logits, dim=-1)

        pred = probabilities.argmax(dim=-1)  # [batch_size]
        output = (logits, pred)

        ## -------loss compute --------
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output += (loss,)
        return output

def mp_matching_func(v1, v2, w, l=20):
    """
    :param v1: (batch, seq_len, hidden_size)
    :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
    :param w: (l, hidden_size)
    :return: (batch, l)
    """
    seq_len = v1.size(1)
    # (1, 1, hidden_size, l)
    w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
    # (batch, seq_len, hidden_size, l)
    v1 = w * torch.stack([v1] * l,
                         dim=3)  # [1,1,hidden_size, l] * [bs, seq_len, hidden_size,l]-> [bs, seq_len, hidden_size,l]
    if len(v2.size()) == 3:
        v2 = w * torch.stack([v2] * l, dim=3)
    else:  # [1,1,hidden_size,l] * [bs,seq_len,hidden_size, l]
        v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * l, dim=3)
    m = F.cosine_similarity(v1, v2,
                            dim=2)  # sim([bs,seq_len,hidden_size, l], [bs,seq_len,hidden_size, l]) ->[bs,seq_len,l]
    return m


def mp_matching_func_pairwise(v1, v2, w, l=20):
    """
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :param w: (l, hidden_size)
    :return: (batch, l, seq_len1, seq_len2)
    """
    # Trick for large memory requirement
    # (1, l, 1, hidden_size)
    w = w.unsqueeze(0).unsqueeze(2)
    # (batch, l, seq_len, hidden_size)
    v1, v2 = w * torch.stack([v1] * l, dim=1), w * torch.stack([v2] * l, dim=1)
    # (batch, l, seq_len, hidden_size->1)
    v1_norm = v1.norm(p=2, dim=3, keepdim=True)
    v2_norm = v2.norm(p=2, dim=3, keepdim=True)
    # (batch, l, seq_len1, seq_len2)
    n = torch.matmul(v1, v2.transpose(2, 3))
    d = v1_norm * v2_norm.transpose(2, 3)
    # (batch, seq_len1, seq_len2, l)
    m = div_with_small_value(n, d).permute(0, 2, 3, 1)
    return m

def attention(v1, v2):
    """
    :param v1: (batch, seq_len1, hidden_size)
    :param v2: (batch, seq_len2, hidden_size)
    :return: (batch, seq_len1, seq_len2)
    """
    # (batch, seq_len1, 1)
    v1_norm = v1.norm(p=2, dim=2, keepdim=True)
    # (batch, 1, seq_len2)
    v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)
    # (batch, seq_len1, seq_len2)
    a = torch.bmm(v1, v2.permute(0, 2, 1))
    d = v1_norm * v2_norm
    return div_with_small_value(a, d)

def div_with_small_value(n, d, eps=1e-8):
    # too small values are replaced by 1e-8 to prevent it from exploding.
    d = d * (d > eps).float() + eps * (d <= eps).float()
    return n / d #[bs, l, sql_len, sql_len]