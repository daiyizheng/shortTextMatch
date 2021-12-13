#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/7 7:14 下午
# @Author : daiyizheng
# @Version：V 0.1
# @File : data_loader.py
# @desc :
import copy
import json
import os
import logging

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from utils import tokenizer

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        sentence1: list. The words of the sequence.
        sentence2: list. The words of the sequence.
        label: (Optional) string. The label of the example.
    """
    def __init__(self, guid, sentence1, sentence2, label):
        self.guid = guid
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_sentence1_ids,
                 input_sentence2_ids,
                 sentence1_attention_mask,
                 sentence2_attention_mask,
                 label_id):
        self.input_sentence1_ids = input_sentence1_ids
        self.input_sentence2_ids = input_sentence2_ids
        self.sentence1_attention_mask = sentence1_attention_mask
        self.sentence2_attention_mask = sentence2_attention_mask
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcessor(object):
    """Processor for the BERT data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def _read_file(cls, input_file, skip_first_line=False):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for i, line in enumerate(f):
                if skip_first_line:
                    if i == 0:
                        continue

                lines.append(line.strip())
            return lines

    @classmethod
    def _read_csv(cls, input_file, set_type):
        """Reads a tab separated value file."""
        df = pd.read_csv(input_file)
        lines = []
        for index, item in df.iterrows():
            if set_type=="test":
                lines.append([index, item["text1"], item["text2"], 0])
            else:
                lines.append([index, item["text1"], item["text2"], item['label']])
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            # id
            id_ = line[0]
            guid = "%s-%s" % (set_type, id_)

            # 1. input_text
            sentence1 = line[1]
            sentence2 = line[2]

            # 标签
            if set_type == "test":
                label = 0
            else:
                label = line[3]

            examples.append(
                InputExample(
                    guid=guid,
                    sentence1=sentence1,
                    sentence2=sentence2,
                    label=label,
                )
            )
        return examples

    def get_examples(self,path, mode):
        """
        Args:
            mode: train, dev, test
        """
        # data_path = os.path.join(self.args.data_dir, "{}.csv".format(mode))
        logger.info("LOOKING AT {}".format(path))
        return self._create_examples(lines=self._read_csv(path, set_type=mode), set_type=mode)

processors = {
    "DSSM": DataProcessor,
    "ESIM":DataProcessor,
    "BIMPM":DataProcessor,
    "ABCNN":DataProcessor,
    "DecomposableAttention":DataProcessor,
    "RE2":DataProcessor,
    "SiaGRU":DataProcessor
}

def convert_examples_to_features(examples,
                                 max_seq_len,
                                 tokenizer,
                                 pad_token_id=0,
                                 unk_token_id=1,
                                 mask_padding_with_zero=True,
                                 vocab_list=None,
                                 special_tokens_count = 0
                                 ):
    pad_token_id = vocab_list.index("[PAD]") if "[PAD]" in vocab_list else pad_token_id
    unk_token_id = vocab_list.index("[UNK]") if "[UNK]" in vocab_list else unk_token_id
    print("pad_token_id", pad_token_id)
    print("unk_token_id", unk_token_id)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        sentenc1 = example.sentence1
        sentenc2 = example.sentence2
        # Account for [CLS] and [SEP]
        sentenc1_tokens = tokenizer(sentenc1)
        sentenc2_tokens = tokenizer(sentenc2)

        if len(sentenc1_tokens) > max_seq_len - special_tokens_count:
            sentenc1_tokens = sentenc1_tokens[:(max_seq_len - special_tokens_count)]

        if len(sentenc2_tokens) > max_seq_len - special_tokens_count:
            sentenc2_tokens = sentenc2_tokens[:(max_seq_len - special_tokens_count)]

        input_sentenc1_ids = [vocab_list.index(w) if w in vocab_list else unk_token_id for w in sentenc1_tokens]
        input_sentenc2_ids = [vocab_list.index(w) if w in vocab_list else unk_token_id for w in sentenc2_tokens]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        sentence1_attention_mask = [1 if mask_padding_with_zero else 0] * len(input_sentenc1_ids)
        sentence2_attention_mask = [1 if mask_padding_with_zero else 0] * len(input_sentenc2_ids)

        # Zero-pad up to the sequence length.
        sentence1_padding_length = max_seq_len - len(sentence1_attention_mask)
        input_sentence1_ids = input_sentenc1_ids + ([pad_token_id] * sentence1_padding_length)
        sentence1_attention_mask = sentence1_attention_mask + ([0 if mask_padding_with_zero else 1] * sentence1_padding_length)

        sentence2_padding_length = max_seq_len - len(sentence2_attention_mask)
        input_sentence2_ids = input_sentenc2_ids + ([pad_token_id] * sentence2_padding_length)
        sentence2_attention_mask = sentence2_attention_mask + ([0 if mask_padding_with_zero else 1] * sentence2_padding_length)

        assert len(input_sentence1_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_sentence1_ids), max_seq_len)
        assert len(sentence1_attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(sentence1_attention_mask), max_seq_len)

        assert len(input_sentence2_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_sentence2_ids), max_seq_len)
        assert len(sentence2_attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(sentence2_attention_mask), max_seq_len)
        label = int(example.label)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("sentenc1 tokens: %s" % " ".join([str(x) for x in sentenc1_tokens]))
            logger.info("sentenc2 tokens: %s" % " ".join([str(x) for x in sentenc2_tokens]))
            logger.info("input_sentence1_ids: %s" % " ".join([str(x) for x in input_sentence1_ids]))
            logger.info("input_sentence2_ids: %s" % " ".join([str(x) for x in input_sentence2_ids]))
            logger.info("sentence1_attention_mask: %s" % " ".join([str(x) for x in sentence1_attention_mask]))
            logger.info("sentence2_attention_mask: %s" % " ".join([str(x) for x in sentence2_attention_mask]))
            logger.info("label: %s (id = %d)" % (example.label, label))


        features.append(
            InputFeatures(input_sentence1_ids=input_sentence1_ids,
                          input_sentence2_ids=input_sentence2_ids,
                          sentence1_attention_mask=sentence1_attention_mask,
                          sentence2_attention_mask=sentence2_attention_mask,
                          label_id=label
                          ))

    return features


def load_and_cache_examples(args, mode, vocab_list=None):
    processor = processors[args.model_name](args)

    # # Load data features from cache or dataset file
    if mode == "train":
        args.data_dir = os.path.dirname(args.train_path)
    elif mode == "dev":
        args.data_dir = os.path.dirname(args.dev_path)
    elif mode == "test":
        args.data_dir = os.path.dirname(args.test_path)
    else:
        raise Exception("mode not in ['train', 'dev', 'test']")

    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            args.model_name,
            args.max_seq_len
        )
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file")
        if mode == "train":
            examples = processor.get_examples(args.train_path, "train")
        elif mode == "dev":
            examples = processor.get_examples(args.dev_path, "dev")
        elif mode == "test":
            examples = processor.get_examples(args.test_path, "test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, vocab_list=vocab_list)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_sentence1_ids = torch.tensor([f.input_sentence1_ids for f in features], dtype=torch.long)
    all_input_sentence2_ids = torch.tensor([f.input_sentence2_ids for f in features], dtype=torch.long)
    all_sentence1_attention_mask = torch.tensor([f.sentence1_attention_mask for f in features], dtype=torch.long)
    all_sentence2_attention_mask = torch.tensor([f.sentence2_attention_mask for f in features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_sentence1_ids,
                            all_input_sentence2_ids,
                            all_sentence1_attention_mask,
                            all_sentence2_attention_mask,
                            all_label_id
                            )
    return dataset

