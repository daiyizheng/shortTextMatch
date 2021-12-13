#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/20 11:36 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : ESIM.py
# @desc :
from sys import platform

import torch.nn as nn
import torch
import numpy as np

class ESIM(nn.Module):
    def __init__(self, args):
        super(ESIM, self).__init__()
        self.args = args

        ## embedding 权重的初始化
        if args.random_init_w2v:
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim)
        else:
            w2v_matrix = np.asarray(args.vector_list)
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim).from_pretrained(
                torch.FloatTensor(w2v_matrix), freeze=False)


        ## ----------  dropout ------------
        if self.args.dropout_rate:
            self.rnn_dropout = RNNDropout(p=self.args.dropout_rate)

        ## ---------------- project layer --------------------
        num_base = 1
        if args.bidirectional:
            num_base = 2
        self.projection = nn.Sequential(nn.Linear(num_base * 4 * args.hidden_size, args.hidden_size), nn.ReLU())

        self.attention = SoftmaxAttention()

        ## -----------encoder-----------------
        self.first_encoder = Seq2SeqEncoder(args, args.embed_dim, args.hidden_size, bidirectional=args.bidirectional)
        self.second_encoder = Seq2SeqEncoder(args, args.hidden_size, args.hidden_size, bidirectional=args.bidirectional)

        ### -------------- classifier layer ------------
        self.classifier = nn.Sequential(nn.Linear(num_base * 4 * args.hidden_size, args.hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(p=args.dropout_rate),
                                            nn.Linear(args.hidden_size, args.hidden_size // 2),
                                            nn.ReLU(),
                                            nn.Dropout(p=args.dropout_rate),
                                            nn.Linear(args.hidden_size // 2, args.num_class))
        ## --------损失函数
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, q1:torch.Tensor, q2:torch.Tensor, labels=None):
        # length of each sentence without padding
        q1_lengths = q1.eq(0).sum(dim=-1)
        q2_lengths = q2.eq(0).sum(dim=-1)

        # calculate the maximum mask length of the current batch
        q1_mask = get_mask(q1, q1_lengths).to(self.args.device)
        q2_mask = get_mask(q2, q2_lengths).to(self.args.device)

        q1_embed = self.word_embedding(q1)  # [batch_size,  max_length, embed_dim]
        q2_embed = self.word_embedding(q2)  # [batch_size,  max_length, embed_dim]

        if self.args.dropout_rate:
            q1_embed = self.rnn_dropout(q1_embed)
            q2_embed = self.rnn_dropout(q2_embed)  ## [bath_size, max_length, embed_dim]

        # 双向lstm编码
        q1_encoded = self.first_encoder(q1_embed, q1_lengths)  # [bs, max_length, hidden_size*bidirection]
        q2_encoded = self.first_encoder(q2_embed, q2_lengths)  # [bs, max_length, hidden_size*bidirection]
        # atention
        q1_aligned, q2_aligned = self.attention(q1_encoded, q1_mask, q2_encoded, q2_mask)  # eij的计算
        # concat
        # [batch_size, q1_batch_max_length, 4*hidden_size]
        q1_combined = torch.cat([q1_encoded, q1_aligned, q1_encoded - q1_aligned, q1_encoded * q1_aligned], dim=-1)
        # [batch_size, q2_batch_max_length, 4*hidden_size]
        q2_combined = torch.cat([q2_encoded, q2_aligned, q2_encoded - q2_aligned, q2_encoded * q2_aligned], dim=-1)
        # 映射层
        projected_q1 = self.projection(q1_combined)  # [batch_size, q1_batch_max_length, hidden_size]
        projected_q2 = self.projection(q2_combined)  # [batch_size, q2_batch_max_length, hidden_size]

        if self.args.dropout_rate:
            projected_q1 = self.rnn_dropout(projected_q1)
            projected_q2 = self.rnn_dropout(projected_q2)

        # 再次经过双向RNN
        q1_compare = self.second_encoder(projected_q1, q1_lengths)  ### [batch_size, q1_batch_max_length, hidden_size] 注意hiodden_size不固定，一般是hidden_dim* direction
        q2_compare = self.second_encoder(projected_q2, q2_lengths)  ### [batch_size, q2_batch_max_length, hidden_size]

        # 平均池化 + 最大池化
        q1_avg_pool = torch.sum(q1_compare * q1_mask.unsqueeze(1).transpose(2, 1), dim=1) / \
                      torch.sum(q1_mask, dim=1,keepdim=True)  ##[batch_size, hidden_size]
        q2_avg_pool = torch.sum(q2_compare * q2_mask.unsqueeze(1).transpose(2, 1), dim=1) / \
                      torch.sum(q2_mask, dim=1,keepdim=True)  ##[batch_size, hidden_size]
        q1_max_pool, _ = replace_masked(q1_compare, q1_mask, -1e7).max(dim=1)  ##[batch_size, hidden_size]
        q2_max_pool, _ = replace_masked(q2_compare, q2_mask, -1e7).max(dim=1)  ##[batch_size, hidden_size]

        # 拼接成最后的特征向量
        merged = torch.cat([q1_avg_pool, q1_max_pool, q2_avg_pool, q2_max_pool], dim=1)  # [batch_size, 4*hidden_size]
        # 分类
        logits = self.classifier(merged)  # [batch_size, num_class]

        ## loss和预测计算
        probabilities = nn.functional.softmax(logits, dim=-1)  # [batch_size, num_class]

        pred = probabilities.argmax(dim=-1)  # [batch_size]
        output = (logits, pred)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output += (loss,)
        ## return (logits, pred, loss)
        return output

def get_mask(sequences_batch, sequences_lengths):
    """

    ARG
    :param sequences_batch:
    :param sequences_lengths:
    :return:
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask

def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
    ARG
    :param batch:
    :param sequences_lengths:
    :param descending:
    :return:
    """
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)  ##  torch.range(start=1, end=6) 的结果是会包含end的，类型：torch.float32  而torch.arange(start=1, end=6)的结果并不包含end 类型：torch.int64
    #idx_range = sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, revese_mapping = sorting_index.sort(0, descending=False) ## 升序
    restoration_index = idx_range.index_select(0, revese_mapping) ## 恢复原来的排序索引
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index # 排序的batch， 排序的序列长度， 排序后的索引， 恢复排序的索引


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).
    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.
    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1]) # [batch_size* q1_batch_max_length, q2_batch_max_length]
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1) # [batch_size, 1, q2_batch_max_length]
    mask = mask.expand_as(tensor).contiguous().float()  # [batch_size, q1_batch_max_length, q2_batch_max_length]
    reshaped_mask = mask.view(-1, mask.size()[-1]) # [batch_size* q1_batch_max_length, q2_batch_max_length]

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1) # [batch_size* q1_batch_max_length, q2_batch_max_length]
    result = result * reshaped_mask #[batch_size* q1_batch_max_length, q2_batch_max_length]
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)  # [batch_size* q1_batch_max_length, q2_batch_max_length]
    return result.view(*tensor_shape)


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied. [bs, max_length, *hidden_size]
        weights: The weights to use in the weighted sum. [bs, q1_len, q2_length]
        mask: A mask to apply on the result of the weighted sum.[bs, max_length]
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor) # tensor-> weights->[batch_size, q1_batch_max_length, q2_batch_max_length]    [batch_size, q2_batch_max_length, hidden_size]   ---> [batch_size,  q1_batch_max_length, hidden_size]

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1) # [batch_size, 1, q1_batch_max_length]
    mask = mask.transpose(-1, -2) # [batch_size, q1_batch_max_length, 1]
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.
    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """
    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.
        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN. [bs, max_length, hidden_size]
                Tensor of size (batch, sequences_length, emebdding_dim).
        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0], sequences_batch.shape[-1]) #[bs, hidden_size]
        dropout_mask = nn.functional.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch #[bs, max_length, hidden_size]


class Seq2SeqEncoder(nn.Module):
    def __init__(self, args, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        "rnn_type must be a class inheriting from torch.nn.RNNBase"
        self.args = args
        super(Seq2SeqEncoder, self).__init__()
        if args.rnn_type.lower() == "lstm":
            self.encoder = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        elif args.rnn_type.lower() == "gru":
            self.encoder = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError("encoder is not nn.RNNBase")

    def forward(self, sequences_batch, sequences_lengths):  # sequences_batch [batch_size, max_length, embed_dim]  sequences_lengths [batch_size]
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        ## 排好序的->降序 sorted_batch [batch_size, max_length, emebd_dim] -> packed_batch [batch_size*per_sample_no_mask_length, embed_dm]
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths.to("cpu"), batch_first=True)
        # outputs->[batch_size*per_sample_no_mask_length, direction*hidden_size]  h_n [direction*num_layer, batch_size, hidden_size] cell_n [direction*num_layer, batch_size, hidden_size]
        outputs, _ = self.encoder(packed_batch, None)
        # outputs->[batch_size, batch_max_length, direction*hidden_size] length> [batch_size]
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # for linux
        if platform == "linux" or platform == "linux2":
            reordered_outputs = outputs.index_select(0, restoration_idx)  ## 恢复排序
        # for win10
        else:
            reordered_outputs = outputs.index_select(0, restoration_idx.long())
        return reordered_outputs

class SoftmaxAttention(nn.Module):
    """
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    The dot product of the encoded vectors in the premises and hypotheses is
    first computed. The softmax of the result is then used in a weighted sum
    of the vectors of the premises for each element of the hypotheses, and
    conversely for the elements of the premises.
    """
    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            premise_mask: A mask for the sequences in the premise batch, to
                ignore padding data in the sequences during the computation of
                the attention.
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, sequences, vector_dim).
            hypothesis_mask: A mask for the sequences in the hypotheses batch,
                to ignore padding data in the sequences during the computation
                of the attention.
        Returns:
            attended_premises: The sequences of attention vectors for the
                premises in the input batch.
            attended_hypotheses: The sequences of attention vectors for the
                hypotheses in the input batch.
        """
        # Dot product between premises and hypotheses in each sequence of the batch.
        #  # contiguous()方法语义上是“连续的”，经常与torch.permute()、torch.transpose()、torch.view()方法一起使用
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous()) ##相似度矩阵 premise_batch[batch_size, batch_max_length, hidden_size] -->  [batch_size, batch_max_length,  batch_max_length]
        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask) # [batch_size, batch_max_length,  batch_max_length]
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)  # [batch_size, batch_max_length,  batch_max_length]
        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        attended_premises = weighted_sum(hypothesis_batch, prem_hyp_attn, premise_mask)# [batch_size, q1_batch_max_length, hidden_size]
        attended_hypotheses = weighted_sum(premise_batch, hyp_prem_attn, hypothesis_mask) # [batch_size, q2_batch_max_length, hidden_size]
        return attended_premises, attended_hypotheses