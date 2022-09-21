import os
import json


from transformers import AutoTokenizer, AutoModel, AutoConfig

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from sklearn.metrics import matthews_corrcoef

import torch
import pandas as pd
import numpy as np

from torch import nn

import time

TOKEN_MAX_LENGTH = {
    'COLA' : 64,
    'WiC' : 256,
    'COPA' : 40,
    'BoolQ' : (400, 80)
} #train, dev ,test, test_labeled
# COPA, COLA는 딱 맞게 잘라놨음. 
#wic는 287 이상이어야 하는데, 일단 아래로. unimodel이니깐, 최대 271+287=558까지.
#boolq는 478 이상이어야 하는데, 일단은 400으로함. question=80이면 됨. (Q는 최대 432까지 가능.)





#individual Metric
def MCC(preds, labels):
    assert len(preds) == len(labels)
    return matthews_corrcoef(labels, preds)

from sklearn.metrics import f1_score, accuracy_score

def F1_scrore():
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    print('f1_score: ', f1_score(y_true, y_pred, average=None))
    print('f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
    
    return f1_score(y_true, y_pred, average='micro')

#monologg/kobert Metric
def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_score(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }



##get parent/home directory path##
def getParentPath(pathStr):
    return os.path.abspath(pathStr+"../../")
#return parentPth/parentPth of pathStr -> hdd1/
def getHomePath(pathStr):
    return getParentPath(getParentPath(getParentPath(pathStr))) #ast/src/
