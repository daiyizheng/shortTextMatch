# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/11/25 13:47
# software: PyCharm

"""
文件说明：
    
"""
import torch.optim as optim

from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)


def make_scheduler(optimizer, decay_name='linear', t_max=None, warmup_steps=None):
    if decay_name == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif decay_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max # max_train_steps
        )
    elif decay_name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max # max_train_steps
        )
    elif decay_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max # max_train_steps
        )
    else:
        raise Exception('Unknown lr scheduler: {}'.format(decay_name))
    return scheduler
