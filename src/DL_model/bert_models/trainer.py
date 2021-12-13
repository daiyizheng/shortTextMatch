#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/25 12:49 上午
# @Author : daiyizheng
# @Version：V 0.1
# @File : trainer.py
# @desc :
import logging
import os, sys
import json

sys.path.insert(0, "./")

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

from src.DL_model.bert_models.configs import MODEL_CLASSES, MODEL_PATH_MAP
from src.DL_model.bert_models.metrics import compute_metrics, correct_predictions
from src.DL_model.bert_models.schedulers import make_scheduler
from src.DL_model.bert_models.optimizers import make_optimizer


logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset



        ## 加载模型
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_name]
        self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                        finetuning_task=args.task,
                                                        gradient_checkpointing=True)

        self.model = self.model_class.from_pretrained(
            args.model_name_or_path,
            # config=self.config,
            args=args,
        )

        self.model.to(args.device)

        # tensorboardx
        self.writer = SummaryWriter(args.tensorboardx_path)

        # for early  stopping
        self.global_epoch = 0
        self.metric_key_for_early_stop = args.metric_key_for_early_stop
        self.best_score = -1e+10
        self.patience = args.patience
        self.early_stopping_counter = 0
        self.do_early_stop = False

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size
        )
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        ## mdoel information
        # Prepare optimizer and schedule (linear warmup and decay)
        print("**********************************Prepare optimizer and schedule start************************")
        for n, p in self.model.named_parameters():
            print(n)
        print("**********************************Prepare optimizer and schedule middle************************")

        # -------------------- Preparation for training  ------------------- #
        # if self.args.model_name=="distilbert":
        #     # 待优化的参数
        #     param_optimizer = list(self.model.named_parameters())
        #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        #     optimizer_grouped_parameters = [
        #         {
        #             'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        #             'weight_decay': 0.01
        #         },
        #         {
        #             'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        #             'weight_decay': 0.0
        #         }
        #     ]
        #     optimizer = AdamW(optimizer_grouped_parameters,
        #                       lr=self.args.lr,
        #                       eps=self.args.adam_epsilon)
        #
        #     # scheduler = get_linear_schedule_with_warmup(
        #     scheduler = get_linear_schedule_with_warmup(
        #         optimizer,
        #         num_warmup_steps=self.args.warmup_steps,
        #         num_training_steps=t_total,
        #     )
        # else:
        optimizer = make_optimizer(args=self.args, model=self.model)
        scheduler = make_scheduler(optimizer, warmup_steps=self.args.warmup_steps, t_max=t_total)
        print("**********************************Prepare optimizer and schedule end***************************")
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            self.global_epoch = self.global_epoch + 1
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            ## 每个epoch的指标
            epoch_running_loss = 0.0
            epoch_correct_preds = 0
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          }
                outputs = self.model(**inputs)
                loss = outputs[-1]
                pred = outputs[1]
                labels = batch[3]

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                epoch_correct_preds += correct_predictions(pred, labels)
                epoch_running_loss += loss.item()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results = self.evaluate(writer=True)

                        logger.info("*" * 50)
                        logger.info("auc score: {}".format(results.get("auc", 0.0)))
                        logger.info("current step score for metric_key_for_early_stop: {}".format(results.get(self.metric_key_for_early_stop, 0.0)))
                        logger.info("best score for metric_key_for_early_stop: {}".format(self.best_score))
                        logger.info("*" * 50)

                        if results.get(self.metric_key_for_early_stop, ) > self.best_score:
                            self.best_score = results.get(self.metric_key_for_early_stop, )
                            self.early_stopping_counter = 0
                            if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                                self.save_model()

                        else:
                            self.early_stopping_counter += 1
                            if self.early_stopping_counter >= self.patience:
                                self.do_early_stop = True

                                logger.info("best score is {}".format(self.best_score))

                        if self.do_early_stop:
                            break

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

                if self.do_early_stop:
                    epoch_iterator.close()
                    break

            #  每个epoch
            avg_loss = epoch_running_loss/len(self.train_dataset)
            avg_acc = epoch_correct_preds/len(self.train_dataset)
            logger.info("**********train epoch:{}, avg_loss:{}, avg_acc:{}**********".format(_+1, avg_loss, avg_acc))
            self.writer.add_scalar("train/loss", avg_loss, self.global_epoch)
            self.writer.add_scalar("train/acc", avg_acc, self.global_epoch)

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

            if self.do_early_stop:
                train_iterator.close()
                break
        self.writer.close()
        return global_step, tr_loss / global_step

    def evaluate(self, writer=False):
        dataset = self.dev_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on dev dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          }

                outputs = self.model(**inputs) # (logits, pred, loss)
                pred = outputs[1] # #[batch_size, num_class]
                tmp_eval_loss = outputs[-1]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # label prediction

            if preds is None:
                preds = pred.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, pred.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # label prediction result
        results_labels = compute_metrics(preds, out_label_ids)
        for key_, val_ in results_labels.items():
            results[key_] = val_

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if writer:
            self.writer.add_scalar("dev/loss", eval_loss, self.global_epoch)
            self.writer.add_scalar("dev/accuracy", results.get("accuracy", 0.0), self.global_epoch)
            self.writer.add_scalar("dev/macro avg__f1-score", results.get("macro avg__f1-score", 0.0), self.global_epoch)
            self.writer.add_scalar("dev/auc", results.get("auc", 0.0), self.global_epoch)
        return results

    def predict(self):
        dataset = self.test_dataset
        test_sample = SequentialSampler(dataset)
        test_dataloader = DataLoader(dataset, sampler=test_sample,batch_size=self.args.eval_batch_size)
        logger.info("***** Running evaluation on test dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        self.model.eval()

        preds = None
        for batch in tqdm(test_dataloader, desc="Prediction"):
            batch = tuple(t.to(self.args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          }
                outputs = self.model(**inputs) # (logits, pre)
                pred = outputs[1]

                if preds is None:
                    preds = pred.detach().cpu().numpy()
                else:
                    preds = np.append(preds, pred.detach().cpu().numpy(), axis=0)

        # label prediction result
        df_test = pd.read_csv(self.args.test_path)
        df_test['label'] = preds
        df_test.to_csv(os.path.join(self.args.model_dir, os.path.basename(self.args.test_path)), index=None, encoding="utf_8_sig")

    def save_model(self):
        # Save models checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        if self.args.model_name == "nezha":
            json.dump(
                model_to_save.config.__dict__,
                open(os.path.join(self.args.model_dir, "config.json"), "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2
            )
            state_dict = model_to_save.state_dict()
            output_model_file = os.path.join(self.args.model_dir, "pytorch_model.bin")
            torch.save(state_dict, output_model_file)

        else:
            model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained models
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving models checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether models exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:

            if self.args.model_name == "nezha":
                output_model_file = os.path.join(self.args.model_dir, "pytorch_model.bin")
                self.model.load_state_dict(torch.load(output_model_file, map_location=self.args.device))

            else:
                self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                              config=self.config,
                                                              args=self.args
                                                              )
            self.model.to(self.args.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some models files might be missing...")