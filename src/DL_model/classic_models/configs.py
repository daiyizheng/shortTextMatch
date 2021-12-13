#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/7 7:27 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : configs.py
# @desc :
from src.DL_model.classic_models.models import DSSM, BIMPM, ABCNN, DecomposableAttention, RE2, SiaGRU, ESIM

MODEL_CLASSES = {
    "DSSM":DSSM,
    "BIMPM":BIMPM,
    "ABCNN":ABCNN,
    "DecomposableAttention":DecomposableAttention,
    "RE2":RE2,
    "SiaGRU":SiaGRU,
    "ESIM":ESIM
}