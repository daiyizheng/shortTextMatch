#!/bin/bash
starttime=`date +'%Y-%m-%d %H:%M:%S'`;
start_seconds=$(date --date="$starttime" +%s);
## vector_size
echo "开始时间: $start_seconds";
echo "--vector_size 100 --epochs 5 --sg 0 --hs 1 --window 1";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 1;
echo "--vector_size 100 --epochs 5 --sg 0 --hs 1 --window 2";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 2;
echo "--vector_size 200 --epochs 5 --sg 0 --hs 1 --window 2";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 200 --epochs 5 --sg 0 --hs 1 --window 2;
echo "--vector_size 300 --epochs 5 --sg 0 --hs 1 --window 2";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 300 --epochs 5 --sg 0 --hs 1 --window 2;
## window
echo "--vector_size 100 --epochs 5 --sg 0 --hs 1 --window 1";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/char --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 1;
echo "--vector_size 100 --epochs 5 --sg 0 --hs 1 --window 3";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 3;
echo "--vector_size 100 --epochs 5 --sg 0 --hs 1 --window 4";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 4;
echo "--vector_size 300 --epochs 5 --sg 0 --hs 1 --window 5";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 5;
echo "--vector_size 300 --epochs 5 --sg 0 --hs 1 --window 6";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 6;
echo "--vector_size 300 --epochs 5 --sg 0 --hs 1 --window 8";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 8;
echo "--vector_size 300 --epochs 5 --sg 0 --hs 1 --window 10";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 10;
echo "--vector_size 300 --epochs 5 --sg 0 --hs 1 --window 12";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 12;
echo "--vector_size 300 --epochs 5 --sg 0 --hs 1 --window 15";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 5 --sg 0 --hs 1 --window 15;
## epochs
echo "--vector_size 100 --epochs 10 --sg 0 --hs 1 --window 2";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 10 --sg 0 --hs 1 --window 2
echo "--vector_size 100 --epochs 15 --sg 0 --hs 1 --window 2";
python train_word2vec.py --corpus_path ./corpus/char/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 15 --sg 0 --hs 1 --window 2;
echo "--vector_size 100 --epochs 20 --sg 0 --hs 1 --window 2";
python train_word2vec.py --corpus_path ./corpus/word/ --model_save_dir ./embedding_vec/word --vector_size 100 --epochs 20 --sg 0 --hs 1 --window 2;
endtime=`date +'%Y-%m-%d %H:%M:%S'`;
end_seconds=$(date --date="$endtime" +%s);
echo "本次运行时间： "$((end_seconds-start_seconds))"s";