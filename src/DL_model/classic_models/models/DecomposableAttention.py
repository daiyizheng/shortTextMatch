#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/20 11:35 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : DecomposableAttention.py
# @desc :

import torch.nn as nn
import torch
import numpy as np

class DecomposableAttention(nn.Module):
    def __init__(self, args):
        super(DecomposableAttention, self).__init__()
        self.args = args

        ## embedding 权重的初始化
        if args.random_init_w2v:
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim)
        else:
            w2v_matrix = np.asarray(args.vector_list)
            self.word_embedding = nn.Embedding(len(args.vocab_list), args.embed_dim).from_pretrained(
                torch.FloatTensor(w2v_matrix), freeze=False)

        ## ---------------- project layer --------------------
        self.project_layer = nn.Linear(args.embed_dim, args.hidden_size1)

        ## ----------------- encoder layer --------------------
        self.F = nn.Sequential(nn.Dropout(args.dropout_rate),
                               nn.Linear(args.hidden_size1, args.hidden_size2),
                               nn.ReLU(),
                               nn.Dropout(args.dropout_rate),
                               nn.Linear(args.hidden_size2, args.hidden_size3),
                               nn.ReLU())

        self.G = nn.Sequential(nn.Dropout(args.dropout_rate),
                               nn.Linear(2 * args.hidden_size1, args.hidden_size2),
                               nn.ReLU(),
                               nn.Dropout(args.dropout_rate),
                               nn.Linear(args.hidden_size2, args.hidden_size3),
                               nn.ReLU())

        self.H = nn.Sequential(nn.Dropout(args.dropout_rate),
                               nn.Linear(2 * args.hidden_size1, args.hidden_size2),
                               nn.ReLU(),
                               nn.Dropout(args.dropout_rate),
                               nn.Linear(args.hidden_size2, args.hidden_size3),
                               nn.ReLU())

        ## ------------------ classifier layer --------------------
        self.classifier = nn.Linear(args.hidden_size3, args.num_class)

        ## 损失函数
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, q1:torch.Tensor, q2:torch.Tensor, labels=None):
        # length of each sentence without padding
        q1_lengths = q1.eq(0).sum(dim=-1)
        q2_lengths = q2.eq(0).sum(dim=-1)

        q1_mask = generate_sent_masks(q1, q1_lengths).to(self.args.device)
        q2_mask = generate_sent_masks(q2, q2_lengths).to(self.args.device)

        q1_embed = self.word_embedding(q1)
        q2_embed = self.word_embedding(q2)
        # q1_embed = self.embeding_dropout(q1_embed)
        # q2_embed = self.embeding_dropout(q2_embed)

        # project_embedd编码
        q1_encoded = self.project_layer(q1_embed)
        q2_encoded = self.project_layer(q2_embed)

        # Attentd
        attend_out1 = self.F(q1_encoded)
        attend_out2 = self.F(q2_encoded)
        similarity_matrix = attend_out1.bmm(attend_out2.transpose(2, 1).contiguous())

        # Softmax attention weights.
        prem_hyp_attn = masked_softmax(similarity_matrix, q2_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(), q1_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        q1_aligned = weighted_sum(q2_encoded, prem_hyp_attn, q1_mask)
        q2_aligned = weighted_sum(q1_encoded, hyp_prem_attn, q2_mask)
        # q1_aligned = weighted_sum(attend_out2, prem_hyp_attn, q1_mask)
        # q2_aligned = weighted_sum(attend_out1, hyp_prem_attn, q2_mask)

        # compare
        compare_i = torch.cat((q1_encoded, q1_aligned), dim=2)  # [bs, seq_len1, 2*hz]
        compare_j = torch.cat((q2_encoded, q2_aligned), dim=2)  # [bs, seq_len2, 2*hz]
        v1_i = self.G(compare_i)  # [bs, seq_len1, hz]
        v2_j = self.G(compare_j)  # [bs, seq_len2, hz]

        # Aggregate (3.3)
        v1_sum = torch.sum(v1_i, dim=1)  # [bs, hz]
        v2_sum = torch.sum(v2_j, dim=1)  # [bs, hz]

        # classifier
        output_tolast = self.H(torch.cat((v1_sum, v2_sum), dim=1))
        logits = self.classifier(output_tolast)

        # output
        probabilities = nn.functional.softmax(logits, dim=-1)
        pred = probabilities.argmax(dim=-1)  # [batch_size]
        output = (logits, pred)

        ## -------loss compute --------
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            output += (loss,)
        return output


def generate_sent_masks(enc_hiddens, source_lengths):
    """ Generate sentence masks for encoder hidden states.
    @param enc_hiddens (Tensor): encodings of shape (b, src_len, h), where b = batch size,
                                 src_len = max source length, h = hidden size.
    @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.len = batch size
    @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                where src_len = max source length, b = batch size.
    """
    enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks

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
    reshaped_tensor = tensor.view(-1, tensor_shape[-1]) #[bs*seq_len1, seq_len2]
    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1) #[bs, 1, seq_len2]
    mask = mask.expand_as(tensor).contiguous().float() #[bs, seq_len1, seq_len2]
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)#[bs, seq_len1, seq_len2]


def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.
    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.
    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor) #[bs, seq_len1, seq_len2] [bs, seq_len2, hz]

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask