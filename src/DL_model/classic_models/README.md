```shell script
export model_name=SiaGRU
python src/DL_model/classic_models/main.py \
  --task LCQMC \
  --model_name ${model_name} \
  --model_dir ./output/LCQMC_${model_name}_20211007 \
  --tensorboardx_path  ./output/logs/runs/LCQMC_${model_name}_20211007  \
  --train_path ./datasets/LCQMC/LCQMC_train.csv \
  --dev_path ./datasets/LCQMC/LCQMC_dev.csv \
  --test_path ./datasets/LCQMC/LCQMC_test.csv \
  --vocab_file ./resources/word2vec/vocab.txt \
  --w2v_file ./resources/word2vec/token_vec_300.bin \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 50 \
  --lr 5e-4 
```

```shell script
#!/bin/bash
time=$(date "+%Y%m%d%H%M%S")
model_name=("DSSM")
task="LCQMC"
for md in ${model_name[*]}
do
  python src/DL_model/classic_models/main.py \
  --task $task \
  --model_name $md \
  --model_dir ./output/"${task}_${md}_${time}" \
  --tensorboardx_path  ./output/logs/runs/"${task}_${md}_${time}"  \
  --train_path ./datasets/LCQMC/LCQMC_train.csv \
  --dev_path ./datasets/LCQMC/LCQMC_dev.csv \
  --test_path ./datasets/LCQMC/LCQMC_test.csv \
  --vocab_file ./resources/word2vec/vocab.txt \
  --w2v_file ./resources/word2vec/token_vec_300.bin \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 50
done
```

可以明显看出训练200轮后结果趋于极端，而这些极端的负面Loss拉大了总体Loss导致验证集Loss飙升。
出现这种情况大多是训练集验证集数据分布不一致，或者训练集过小，未包含验证集中所有情况，也就是过拟合导致的。
而解决这种现象可以尝试以下几种策略：
增加训练样本
增加正则项系数权重，减小过拟合
加入早停机制，ValLoss上升几个epoch直接停止
采用Focal Loss
加入Label Smoothing

不过个人感觉主要还是增加训练样本比较靠谱..而且不用太关心ValLoss，关注下ValAccuracy就好。
我的这个实验虽然只训练10个epoch在验证集上的准确率高且ValLoss小，
但在测试集上结果是巨差的，而训练200个epoch的模型ValLoss虽然巨高但测试集效果还不错。
平均倒数排名