import wandb
import argparse

import torch
import numpy as np
import random
from torch import nn

import copy


def set_seed(seedNum, device):
    torch.manual_seed(seedNum)
    torch.cuda.manual_seed(seedNum)
    torch.cuda.manual_seed_all(seedNum) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seedNum)
    random.seed(seedNum)

from torch.utils.data import DataLoader

def load_data(dataset_train, dataset_dev, dataset_test_label, batch_size):
    TrainLoader = DataLoader(dataset_train, batch_size = batch_size)
    EvalLoader = DataLoader(dataset_dev, batch_size = batch_size)
    InferenceLoader = DataLoader(dataset_test_label, batch_size = batch_size)

    return TrainLoader, EvalLoader, InferenceLoader


from torch.optim import AdamW, SGD

def build_optimizer(parameters, lr, weight_decay, type):
    if type == "AdamW":
        optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay, eps=1e-8)
    elif type == "SGD":
        optimizer = SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    return optimizer


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes = 2, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def evaluation_f1(true_data, pred_data):

    true_data_list = true_data
    pred_data_list = pred_data

    ce_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    pipeline_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    for i in range(len(true_data_list)):

        # TP, FN checking
        is_ce_found = False
        is_pipeline_found = False
        for y_ano  in true_data_list[i]['annotation']:
            y_category = y_ano[0]
            y_polarity = y_ano[2]

            for p_ano in pred_data_list[i]['annotation']:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is True:
                ce_eval['TP'] += 1
            else:
                ce_eval['FN'] += 1

            if is_pipeline_found is True:
                pipeline_eval['TP'] += 1
            else:
                pipeline_eval['FN'] += 1

            is_ce_found = False
            is_pipeline_found = False

        # FP checking
        for p_ano in pred_data_list[i]['annotation']:
            p_category = p_ano[0]
            p_polarity = p_ano[1]

            for y_ano  in true_data_list[i]['annotation']:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is False:
                ce_eval['FP'] += 1

            if is_pipeline_found is False:
                pipeline_eval['FP'] += 1


    ce_precision = 0 if (ce_eval['TP']+ce_eval['FP']) == 0 else ce_eval['TP']/(ce_eval['TP']+ce_eval['FP'])
    ce_recall = 0 if (ce_eval['TP']+ce_eval['FN']) == 0 else ce_eval['TP']/(ce_eval['TP']+ce_eval['FN'])

    ce_result = {
        'Precision': ce_precision,
        'Recall': ce_recall,
        'F1': 0 if (ce_recall+ce_precision) == 0 else 2*ce_recall*ce_precision/(ce_recall+ce_precision)
    }

    pipeline_precision = 0 if (pipeline_eval['TP']+pipeline_eval['FP']) == 0 else pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FP'])
    pipeline_recall = 0 if (pipeline_eval['TP']+pipeline_eval['FN']) == 0 else pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FN'])

    pipeline_result = {
        'Precision': pipeline_precision,
        'Recall': pipeline_recall,
        'F1': 0 if (pipeline_recall+pipeline_precision) == 0 else 2*pipeline_recall*pipeline_precision/(pipeline_recall+pipeline_precision)
    }

    return {
        'category extraction result': ce_result,
        'entire pipeline result': pipeline_result
    }