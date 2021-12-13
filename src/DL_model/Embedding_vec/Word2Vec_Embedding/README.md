# 该文件主要是使用Word2Vec来构建向量的
## 基于字的向量
```shell script
python src/DL_model/Embedding_vec/Word2Vec_Embedding/train_word2vec.py \
--corpus_path ./datasets/word2vec_corpus/char/ \
--model_save_dir ./resources/QYTM_word2vec/ \
--vector_size 300 \
--epochs 5 \
--sg 0 \
--hs 1 \
--window 5;
```
