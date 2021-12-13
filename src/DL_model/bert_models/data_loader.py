#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/25 12:49 上午
# @Author : daiyizheng
# @Version：V 0.1
# @File : data_loader.py
# @desc :

import copy
import json
import os
import logging
import math

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import TensorDataset


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

    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels

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
        for index, item in tqdm(df.iterrows()):
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
    'bert': DataProcessor,
    'albert':DataProcessor,
    'roberta':DataProcessor,
    'distilbert':DataProcessor,
    'xlnet':DataProcessor,
    'electra':DataProcessor,
    'nezha':DataProcessor
}

def convert_examples_to_features(examples,
                                 max_seq_len,
                                 tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 special_tokens_count=0
                                 ):
    # Setting based on the current models type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize
        sentence1_tokens = tokenizer.tokenize(example.sentence1)
        sentence2_tokens = tokenizer.tokenize(example.sentence2)
        # print(tokens, " ".join(example.words), )

        # Account for [CLS] and [SEP]
        if len(sentence1_tokens) > (max_seq_len - special_tokens_count)//2:
            sentence1_tokens = sentence1_tokens[:(max_seq_len - special_tokens_count)//2]

        if len(sentence2_tokens) > (max_seq_len - special_tokens_count)//2:
            sentence2_tokens = sentence2_tokens[:(max_seq_len - special_tokens_count)//2]

        # Add [SEP] token
        tokens = sentence1_tokens + [sep_token] + sentence2_tokens + [sep_token]
        token_type_ids = [sequence_a_segment_id] * (len(sentence1_tokens)+1) + [1]*(len(sentence2_tokens)+1)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = int(example.label)

        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label_id: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=label_id,
            )
        )

    return features


def load_and_cache_examples(args,tokenizer, mode):
    processor = processors[args.model_name](args)

    # # level 标签的频次
    # label2freq = json.load(
    #     open(args.label2freq_level_dir, "r", encoding="utf-8"),
    # )

    # 加载label list
    # label_list_level = get_labels(args.label_file_level_dir)

    # # Load data features from cache or dataset file
    if mode=="train":
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
            args.model_name_or_path.split("/")[-1],
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples(args.train_path, mode)
        elif mode == "dev":
            examples = processor.get_examples(args.dev_path, mode)
        elif mode == "test":
            examples = processor.get_examples(args.test_path, mode)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        features = convert_examples_to_features(
            examples,
            args.max_seq_len,
            tokenizer,
            special_tokens_count=3)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels,
    )

    return dataset
