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
from soynlp.normalizer import *


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


from sklearn.model_selection import StratifiedKFold, KFold


def kFold (file_list, n_splits, which_k):
    
    kf = KFold(n_splits = n_splits, shuffle=True, random_state=1)
    
    data = jsonlload(file_list)
    
    data = np.array(data)
    
    n_iter = 0
    
    for train_index, test_index in kf.split(data):
        
        print("test : ", test_index)
        
        n_iter += 1
        if n_iter == which_k:
            features_train = data[train_index]
            features_test = data[test_index]
            print(f'------------------ {n_splits}-Fold 중 {n_iter}번째 ------------------')
            
            return features_train, features_test

def custom_stratified_KFold(file_list, n_splits, which_k, label_name):

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True)

    data = pd.DataFrame()
    
    for data_file in file_list:
        data = pd.concat([data, pd.read_csv(os.path.join(datasetPth, data_file), sep="\t")])

    features = data.iloc[:,:]

    label = data[label_name]

    n_iter = 0

    for train_idx, test_idx in skf.split(features, label):
        n_iter += 1

        # label_train = label.iloc[train_idx]
        # label_test = label.iloc[test_idx]

        features_train = features.iloc[train_idx]
        features_test = features.iloc[test_idx]

        if n_iter == which_k:
            print(f'------------------ {n_splits}-Fold 중 {n_iter}번째 ------------------')

            # print(features_test)

            return features_train, features_test