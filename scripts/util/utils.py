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
import re

from sklearn.metrics import f1_score, accuracy_score

def F1_scrore():
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    print('f1_score: ', f1_score(y_true, y_pred, average=None))
    print('f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
    
    return f1_score(y_true, y_pred, average='micro')


# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        # json.dump(j, f, ensure_ascii=False)
        for i in j:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
