#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/11/25 12:49 上午
# @Author : daiyizheng
# @Version：V 0.1
# @File : metrics.py
# @desc :
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

def correct_predictions(preds, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    # _, out_classes = output_probabilities.max(dim=1)
    correct = (preds == targets).sum()
    return correct.item()

def compute_metrics(intent_preds, intent_labels):
    """
    evaluation metrics
    :param intent_preds: prediction labels
    :param intent_labels:glod labels
    :return:dict {}
    """
    assert len(intent_preds) == len(intent_labels)
    results = {}
    classification_report_dict = classification_report(intent_labels, intent_preds, output_dict=True)
    auc = roc_auc_score(intent_labels, intent_preds)
    results["auc"] = auc
    for key0, val0 in classification_report_dict.items():
        if isinstance(val0, dict):
            for key1, val1 in val0.items():
                results[key0 + "__" + key1] = val1

        else:
            results[key0] = val0
    return results