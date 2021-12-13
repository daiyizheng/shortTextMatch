#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/22 11:55 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : cosineSimilarity.py
# @desc :
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def cosine_sim(input_vec, candidate_vec):
    results = list(cosine_similarity(csr_matrix(input_vec), csr_matrix(candidate_vec))[0])
    return np.array(results)

