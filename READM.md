# 文本匹配
## 依赖
```shell script
pip install transformers==4.5.1
```

```shell script
export task=QYTM
export model_name=RE2
CUDA_VISIBLE_DEVICES=1 python src/DL_model/${model_name}/main.py \
  --task ${task} \
  --model_name ${model_name} \
  --output_dir ./output/${task}_${model_name}_20211007 \
  --tensorboardx_path  ./output/logs/runs/${task}_${model_name}_20211007  \
  --train_path ./datasets/${task}/train.csv \
  --dev_path ./datasets/${task}/dev.csv \
  --test_path ./datasets/${task}/test.csv \
  --label_file_level_dir ./datasets/${task}/labels_level.txt \
  --label2freq_level_dir ./datasets/${task}/label2freq_level.json \
  --vocab_file ./resources/word2vec/vocab.txt \
  --w2v_file ./resources/word2vec/token_vec_300.bin \
  --train_batch_size 256 \
  --eval_batch_siz 256 \
  --do_train \
  --do_eval \
  --do_predict \
  --num_train_epochs 50
```

<table>
        <tr>
            <th>model</th>
            <th>LCQMC(f1-score)</th>
        </tr>
        <tr>
            <th>DSSM</th>
            <th>0.7114</th>
        </tr>
        <tr>
            <th>ABCNN</th>
            <th>0.7743</th>
        </tr>
        <tr>
            <th>BIMPM</th>
            <th>0.8590</th>
        </tr>
        <tr>
            <th>DecomposableAttention</th>
            <th>0.7526</th>
        </tr>
        <tr>
            <th>ESIM</th>
            <th>0.8573</th>
        </tr>
        <tr>
            <th>RE2</th>
            <th>0.8182</th>
        </tr>
        <tr>
            <th>SiaGRU</th>
            <th>0.8302</th>
        </tr>
        <tr>
            <th>Bert</th>
            <th>0.8817</th>
        </tr>
        <tr>
            <th>RoBerta</th>
            <th>0.8897</th>
        </tr>
        <tr>
            <th>XlNet</th>
            <th>0.7828</th>
        </tr>
        <tr>
            <th>ELECTRA</th>
            <th>1</th>
        </tr>
        <tr>
            <th>DistilBert</th>
            <th>？？</th>
        </tr>
        <tr>
            <th>AlBert</th>
            <th>？？</th>
        </tr>
        <tr>
            <th>NEZHA</th>
            <th>1</th>
        </tr>
    </table>
