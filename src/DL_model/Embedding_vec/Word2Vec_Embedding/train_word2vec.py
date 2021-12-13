# -*- coding: utf-8 -*-

import os

from gensim.models import Word2Vec
import argparse
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            print(fname)
            for line in open(os.path.join(self.dirname, fname), "r", encoding="utf-8"):
                yield line.split()


def train_w2v_model(args, sentences: MySentences, ):
    print('Start...')
    model = Word2Vec(
        sentences,
        sg=args.sg,
        hs=args.hs,
        vector_size=args.vector_size,
        window=args.window,
        min_count=1,
        workers=2,
        epochs=args.epochs,
    )

    vocab2id = model.wv.key_to_index
    vocab_list = [k for k in vocab2id]
    with open(os.path.join(args.model_save_dir, f"epoch_{args.epochs}_window_{args.window}_sg_{args.sg}_hs_{args.hs}_dim_{args.vector_size}_vocab.txt"), "w", encoding="utf-8") as fw:
        fw.write("\n".join(vocab_list))
    model.wv.save_word2vec_format(os.path.join(args.model_save_dir, f"epoch_{args.epochs}_window_{args.window}_sg_{args.sg}_hs_{args.hs}_dim_{args.vector_size}_w2v.vectors"), binary=False)
    print("Finished!")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--corpus_path", default="./corpus/char/", type=str, required=True, help="corpus path")
    parse.add_argument("--model_save_dir", default="./embedding_vec/char", type=str, required=True, help="Embedding Save Path")
    parse.add_argument("--vector_size", default=100,  type=int, required=True, choices=[100,200,300], help="")
    parse.add_argument("--epochs", default=5, type=int, required=True, choices=[5,10,15,20], help="")
    parse.add_argument("--sg", default=0, type=int, required=True, choices=[0,1], help="")  # 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
    parse.add_argument("--hs", default=1, type=int, required=True, choices=[0,1], help="")  # 5) hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。6) negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。
    parse.add_argument("--window", default=2, type=int, required=True, help="", choices=[1,2,3,4,5,6,8,10,12,15])

    args = parse.parse_args()

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    sentences = MySentences(args.corpus_path)
    # sentences = MySentences("./datasets/无监督数据/jsonline")

    train_w2v_model(args, sentences)




