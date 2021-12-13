#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/21 10:35 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : BM25.py
# @desc :
import pickle
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer,_document_frequency, CountVectorizer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
import numpy as np
import scipy.sparse as sp

from src.ML_model.utils.cosineSimilarity import cosine_sim


class Bm25Vectorizer(CountVectorizer):
    def __init__(self,k=1.2,b=0.75, norm="l2", use_idf=True, smooth_idf=True,sublinear_tf=False,*args,**kwargs):
        super(Bm25Vectorizer,self).__init__(*args,**kwargs)
        self._tfidf = Bm25Transformer(k=k,b=b,norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    @property
    def k(self):
        return self._tfidf.k

    @k.setter
    def k(self, value):
        self._tfidf.k = value

    @property
    def b(self):
        return self._tfidf.b

    @b.setter
    def b(self, value):
        self._tfidf.b = value

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.
        """
        X = super(Bm25Vectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.
        """
        X = super(Bm25Vectorizer, self).fit_transform(raw_documents)
        self._tfidf.fit(X)
        return self._tfidf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        """Transform documents to document-term matrix.
        """
        check_is_fitted(self, '_tfidf', 'The tfidf vector is not fitted')

        X = super(Bm25Vectorizer, self).transform(raw_documents)
        return self._tfidf.transform(X, copy=False)

class Bm25Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, k=1.2, b=0.75, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.k = k
        self.b = b
        ##################以下是TFIDFtransform代码##########################
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        _X = X.toarray()
        self.avdl = _X.sum() / _X.shape[0]  # 句子的平均长度
        # print("原来的fit的数据：\n",X)

        # 计算每个词语的tf的值
        self.tf = _X.sum(0) / _X.sum()  # [M] #M表示总词语的数量
        self.tf = self.tf.reshape([1, self.tf.shape[0]])  # [1,M]
        # print("tf\n",self.tf)
        ##################以下是TFIDFtransform代码##########################
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                                        n=n_features, format='csr')

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        ########### 计算中间项  ###############
        cur_tf = np.multiply(self.tf, X.toarray())  # [N,M] #N表示数据的条数，M表示总词语的数量
        norm_lenght = 1 - self.b + self.b * (X.toarray().sum(-1) / self.avdl)  # [N] #N表示数据的条数
        norm_lenght = norm_lenght.reshape([norm_lenght.shape[0], 1])  # [N,1]
        middle_part = (self.k + 1) * cur_tf / (cur_tf + self.k * norm_lenght)
        ############# 结算结束  ################

        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.floating):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1
        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        ############# 中间项和结果相乘  ############
        X = X.toarray() * middle_part
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)
        ############# #########

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def idf_(self):
        ##################以下是TFIDFtransform代码##########################
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))


class BM25Model(object):
    def __init__(self, documents, stop_words, tokenizer):
        self.documents = documents
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.texts = self.pretreatment(documents, stop_words, tokenizer)
        self.vectorizer = Bm25Vectorizer()

    def train(self):
        return self.vectorizer.fit_transform(self.texts)

    def process(self, data):
        self.pretreatment(data, self.stop_words, self.tokenizer)
        return self.vectorizer.transform(data)

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


    def cosine(self, text: str, candidate_text: List, vectorizer):
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

if __name__ == '__main__':
    # format_weibo(word=False)
    # format_xiaohuangji_corpus(word=True)
    import jieba


    # 1. 原始数据
    documents = [
        'hello world',
        'oh hello there',
        'Play it',
        'Play it again Sam,24343,123',
    ]
    tokenizer = jieba.lcut
    stop_words = ["123",]
    obj = BM25Model(documents=documents, stop_words=stop_words, tokenizer=tokenizer)
    # 2. 原始数据向量化
    obj.train()
    obj.save("../../../outputs/bm25.pkl")

    a = obj.cosine("hello there", ['hello world','oh hello there','Play it',], obj.load("../../../outputs/bm25.pkl"))
    print(a)





