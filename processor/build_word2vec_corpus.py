#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/10/4 9:28 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : build_word2vec_corpus.py
# @desc :
import re

def tokenizer(text):
    """
    decs 分词，我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
         注意这种分词会会去除一些字符如（ -
    :param text str: xxxxx
    :return:
    """
    regEx = re.compile('[\\W]+')#我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')#[\u4e00-\u9fa5]中文范围
    sentences = regEx.split(text)
    str_list = []
    for sentence in sentences:
        if res.split(sentence) == None:
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

input_path = "../datasets/word2vec_corpus/corpus"
save_path = "../datasets/word2vec_corpus/char/char_corpus"

fw = open(save_path, "w", encoding="utf-8")
with open(input_path, "r", encoding="utf-8") as fr:
    for line in fr.readlines():
        text = line.strip()
        if text:
            tokens = tokenizer(text)
            fw.write(" ".join(tokens))
            fw.write("\n")

fw.close()




