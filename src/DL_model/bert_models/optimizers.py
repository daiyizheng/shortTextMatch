# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/11/25 13:47
# software: PyCharm

"""
文件说明：
    
"""
import math

import torch
from torch.optim.optimizer import Optimizer
from transformers import AdamW

class Lamb(Optimizer):
    # Reference code: https://github.com/cybertronai/pytorch-lamb
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0,
            clamp_value: float = 10,
            adam: bool = False,
            debias: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss

def get_optimizer_params(args, model):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = 5e-5
    no_decay = ['bias', 'gamma', 'beta']
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    parameters = []
    if args.model_name=="RoBERTa":
        parameters = model.roberta.named_parameters()
    elif args.model_name=="BERT":
        parameters = model.bert.named_parameters()
    else:
        raise ValueError("模型选择错误")
    optimizer_parameters = [
        {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.01},
        {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.01, 'lr': learning_rate/2.6},
        {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.01, 'lr': learning_rate},
        {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.01, 'lr': learning_rate*2.6},
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': learning_rate/2.6},
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': learning_rate*2.6},
        {'params': [p for n, p in parameters if "roberta" not in n], 'lr':1e-3, "momentum" : 0.99},
    ]
    return optimizer_parameters

def make_optimizer_with_layer_type(args, model, optimizer_name="AdamW"):
    optimizer_grouped_parameters = get_optimizer_params(args, model)
    kwargs = {
        'lr':5e-5,
        'weight_decay':0.01,
        # 'betas': (0.9, 0.98),
        # 'eps': 1e-06
    }
    if optimizer_name == "LAMB":
        optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "Adam":
        from torch.optim import Adam
        optimizer = Adam(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
        return optimizer
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer_name))

def make_optimizer_with_model_type(args, model):
    optimizer_grouped_parameters = []
    # embedding部分
    if args.model_name == 'xlnet':
        embeddings_params = list(model.bert.named_parameters()) # 'mask_emb', 'word_embedding'
        embeddings_params = [em_p for em_p in embeddings_params if em_p[0] in ['mask_emb', 'word_embedding.weight']]
    else:
        embeddings_params = list(model.bert.embeddings.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {'params': [p for n, p in embeddings_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         "lr": args.embeddings_learning_rate,
         },
        {'params': [p for n, p in embeddings_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         "lr": args.embeddings_learning_rate,
         }
    ]

    # encoder + bert_pooler 部分
    if args.model_name == "xlnet":
        encoder_params = list(model.bert.named_parameters())
        encoder_params = [em_p for em_p in encoder_params if "layer" in em_p[0]]
    elif args.model_name == "distilbert":
        encoder_params = list(model.bert.transformer.named_parameters())
    else:
        encoder_params = list(model.bert.encoder.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         "lr": args.encoder_learning_rate,
         },
        {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         "lr": args.encoder_learning_rate,
         }
    ]

    # linear层 + 初始化的aggregator部分
    classifier_params = list(model.classifier.named_parameters())

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters += [
        {'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         "lr": args.classifier_learning_rate,
         },
        {'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         "lr": args.classifier_learning_rate,
         }
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.lr,
                      eps=args.adam_epsilon)
    return optimizer


def make_optimizer(args, model):
    model = (
        model.module if hasattr(model, "module") else model
    )
    return make_optimizer_with_model_type(args, model)