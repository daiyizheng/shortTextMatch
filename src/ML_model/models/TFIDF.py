#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/22 11:44 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : TFIDF.py
# @desc :

from typing import List

import jieba
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.ML_model.utils.cosineSimilarity import cosine_sim

class TfidfModel:
    def __init__(self,
                 documents,
                 stop_words,
                 tokenizer,
                 analyzer='word'):
        self.documents = documents
        self.texts = self.pretreatment(documents, stop_words, tokenizer)
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.vectorizer = TfidfVectorizer(stop_words=stop_words,
                                          tokenizer=None,
                                          analyzer=analyzer)

    def save(self, feature_path):
        with open(feature_path, "wb") as fw:
            pickle.dump(self.vectorizer, fw)


    def load(self, feature_path):
        vectorizer = pickle.load(open(feature_path, 'rb'))
        return vectorizer

    @staticmethod
    def pretreatment(texts, stop_words, tokenizer):
        texts = [[word for word in tokenizer(text.lower()) if word not in stop_words] for text in texts]
        return [" ".join(text) for text in texts]

    def train(self):
        return self.vectorizer.fit_transform(self.texts)

    def process(self, data):
        return self.vectorizer.transform(data)

    def cosine(self, text:str, candidate_text:List, vectorizer):
        ## 预处理
        candidate_text = self.pretreatment(candidate_text, self.stop_words, self.tokenizer)
        text = self.pretreatment([text], self.stop_words, self.tokenizer)
        text_vec = vectorizer.transform(text).toarray()[0]
        candidate_vec = []
        for t in candidate_text:
            candidate_vec.append(vectorizer.transform([t]).toarray()[0])
        candidate_vec = np.array(candidate_vec)
        result = cosine_sim(text_vec, candidate_vec)
        return result

from gensim import corpora, models, similarities

class gensimTFIDFModel(object):
    def __init__(self, documents:List, stop_words:List, tokenizer):
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.documents = documents
        self.texts = self.pretreatment(documents,stop_words,tokenizer)
        self.dictionary = corpora.Dictionary(self.texts)


    def train(self):
        corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.tfidf = models.TfidfModel(corpus)
        return self.tfidf, corpus

    def process(self, data:List):
        texts = self.pretreatment(data, self.stop_words, self.tokenizer)
        texts = [self.dictionary.doc2bow(text) for text in texts]
        return self.tfidf[texts]

    @staticmethod
    def pretreatment(documents:List, stop_words:List, tokenizer):
        return [[word for word in tokenizer(text.lower()) if word not in stop_words] for text in documents]

    def save(self, feature_path):
        self.tfidf.save(feature_path)


    def load(self, feature_path):
        return models.TfidfModel.load(feature_path)

    def cosine(self, text:str, candidate_text:List, vectorizer):
        text = self.pretreatment([text], self.stop_words, self.tokenizer)
        candidate_text = self.pretreatment(candidate_text, self.stop_words, self.tokenizer)
        text = [self.dictionary.doc2bow(text) for text in text]
        candidate_text = [self.dictionary.doc2bow(text) for text in candidate_text]
        index = similarities.SparseMatrixSimilarity(vectorizer[candidate_text], num_features=1000)
        sims = [index[new_vec_tfidf] for new_vec_tfidf in text]
        return sims


if __name__ == '__main__':

    corpus = ["自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。",
              "它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。",
              "自然语言处理是一门融语言学、计算机科学、数学于一体的科学。",
              "因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，",
              "所以它与语言学的研究有着密切的联系，但又有重要的区别。",
              "自然语言处理并不是一般地研究自然语言，",
              "而在于研制能有效地实现自然语言通信的计算机系统，",
              "特别是其中的软件系统。因而它是计算机科学的一部分。",
              "北京你好"]

    input_text = "而在于研制能有效地实现"
    candidate_text = ["因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，",
                      "所以它与语言学的研究有着密切的联系，但又有重要的区别。",
                      "自然语言处理并不是一般地研究自然语言，",
                      "而在于研制能有效地实现自然语言通信的计算机系统，",
                      "特别是其中的软件系统。因而它是计算机科学的一部分。",
                      "北京你好"]
    stop_word = ["。", "，"]
    tokenizer = jieba.lcut
    ## ---------------------------------------  TfidfModel -------------------------------------
    obj = TfidfModel(corpus, stop_words=stop_word, tokenizer=tokenizer)
    obj.train()
    obj.save("../../../outputs/tfidf.pkl")
    res = obj.cosine(input_text,
               candidate_text,
               obj.load("../../../outputs/tfidf.pkl"))
    print(res)
    print(res.argsort()[::-1])

    ## ------------------------------ gensimTFIDFModel ---------------------------------------
    obj = gensimTFIDFModel(corpus, stop_words=stop_word, tokenizer=tokenizer)
    obj.train()
    obj.save("../../../outputs/gensimtfidf.pkl")
    obj.load("../../../outputs/gensimtfidf.pkl")

    a = obj.cosine(input_text, candidate_text, obj.load("../../../outputs/gensimtfidf.pkl"))
    print(a[0])
    print(a[0].argsort()[::-1])




