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



from sklearn.model_selection import StratifiedKFold

entity_property_pair = [
    
    '본품#품질', '제품 전체#일반', '제품 전체#품질', '본품#일반', '제품 전체#디자인',
    '본품#편의성', '제품 전체#편의성', '제품 전체#인지도', '패키지/구성품#디자인', '브랜드#일반',
    '제품 전체#가격', '패키지/구성품#편의성', '패키지/구성품#일반', '본품#다양성', '본품#디자인',
    '브랜드#품질', '패키지/구성품#품질', '브랜드#인지도', '브랜드#가격', '패키지/구성품#다양성',
    '제품 전체#다양성', '본품#가격', '브랜드#디자인', '패키지/구성품#가격', '본품#인지도'
    
]   # 분포도 순서

# def custom_stratified_KFold(file_list, n_splits, which_k):
    
#     def jsonlload(fname_list, encoding="utf-8"):
#         json_list = []

#         for index, value in enumerate(fname_list):
#             fname = "../dataset/" + value

#             with open(fname, encoding=encoding) as f:
#                 for line in f.readlines():
#                     # print(line)
#                     json_list.append(json.loads(line))

#         return json_list

#     data = jsonlload(file_list)
#     label = []
    
#     for d in data:
        
#         annotation = d["annotation"]
        
#         ano_index = entity_property_pair.index(annotation[0][0])
        
#         for a in annotation:
#             if entity_property_pair.index(a[0]) > ano_index:
#                 ano_index = entity_property_pair.index(a[0])
        
#         label.append(ano_index)


#     skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=1)
    
#     n_iter = 0

#     for train_idx, test_idx in skf.split(data, label):
#         n_iter += 1

#         # print(test_idx)

#         if n_iter == which_k:
#             print(f'------------------ {n_splits}-Fold 중 {n_iter}번째 ------------------')
            
#             features_train = [data[i] for i in train_idx]
#             features_test = [data[i] for i in test_idx]

#             return features_train, features_test
        
        

def custom_stratified_KFold(file_list, n_splits, which_k):
    
    # jsonl 파일 읽어서 list에 저장
    def jsonlload(fname_list, encoding="utf-8"):
        json_list = []

        for index, value in enumerate(fname_list):
            fname = "../dataset/" + value

            with open(fname, encoding=encoding) as f:
                for line in f.readlines():
                    # print(line)
                    json_list.append(json.loads(line))

        return json_list

    if which_k == 1:
        train_file_list = ["2Fold.jsonl", "3Fold.jsonl"]
        dev_file_list = ["1Fold.jsonl"]
    elif which_k == 2:
        train_file_list = ["1Fold.jsonl", "3Fold.jsonl"]
        dev_file_list = ["2Fold.jsonl"]
    elif which_k == 3:
        train_file_list = ["1Fold.jsonl", "2Fold.jsonl"]
        dev_file_list = ["3Fold.jsonl"]
        
        
        
    augmentation_file_list = ["aug1.json"]
    
    train_file_list = train_file_list + augmentation_file_list

    train_data = jsonlload(train_file_list)
    
    
    dev_file_list = ["data_new.jsonl"]  # don't touch
    
    dev_data = jsonlload(dev_file_list)
    
    return train_data, dev_data