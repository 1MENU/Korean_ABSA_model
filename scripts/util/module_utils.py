import wandb
import argparse

import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
import os

import copy

from util.utils import *

datasetPth = '../dataset/'
saveDirPth_str = "../materials/saved_model/"
predPth = '../materials/pred/'
submissionPth = '../materials/submission/'

def make_directories(task):
    if not os.path.exists(saveDirPth_str + task + "/"):
        os.makedirs(saveDirPth_str + task + "/")
        
    if not os.path.exists(predPth + task + "/"):
        os.makedirs(predPth + task + "/")
        
    if not os.path.exists(submissionPth):
        os.makedirs(submissionPth)


def set_seed(seedNum, device):
    torch.manual_seed(seedNum)
    torch.cuda.manual_seed(seedNum)
    torch.cuda.manual_seed_all(seedNum) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seedNum)
    random.seed(seedNum)

from torch.utils.data import DataLoader

from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, ReduceLROnPlateau, CyclicLR

# build various type of optimizers
def build_optimizer(parameters, lr, weight_decay, type):
    if type == "AdamW":
        optimizer = AdamW(parameters, lr=lr, eps=1e-8)  # weight_decay=weight_decay, 여기서는 분리하니까 X
    elif type == "SGD":
        optimizer = SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    return optimizer

def build_scheduler(optimizer, name):
    if name == "ExponentialLR":
        scheduler = ExponentialLR(optimizer)
    elif name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer)
    elif name == "CyclicLR":
        scheduler = CyclicLR(optimizer)
    elif name == "None":
        scheduler = False
        
    return scheduler  

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


class FocalLossWithSmoothing(nn.Module):
    def __init__(
            self,
            num_classes: int,
            gamma: int = 1,
            lb_smooth: float = 0.1,
            size_average: bool = True,
            ignore_index: int = None,
            alpha: float = None):
        """
        :param gamma:
        :param lb_smooth:
        :param ignore_index:
        :param size_average:
        :param alpha:
        """
        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._size_average = size_average
        self._ignore_index = ignore_index
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._alpha = alpha

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')
        if self._alpha is not None:
            if self._alpha <= 0 or self._alpha >= 1:
                raise ValueError('Alpha must be 0 <= alpha <= 1')

    def forward(self, logits, label):
        """
        :param logits: (batch_size, class, height, width)
        :param label:
        :return:
        """
        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = 1. - self._lb_smooth, self._lb_smooth / (self._num_classes - 1)
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits, dim=-1)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level
    

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

    # print(ce_eval)

    return {
        'category extraction result': ce_result,
        'entire pipeline result': pipeline_result
    }



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

def stratified_KFold(file_list, n_splits, which_k, label_name):

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

import pprint

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

def jsonltoDataFrame(fname_list, encoding="utf-8"):
    df = pd.DataFrame()

    for index, value in enumerate(fname_list):
        print(value['annotation'])
        fname = value
        idf = pd.read_json(fname, lines=True)
        df = pd.concat([df, idf],ignore_index=True)
        
    #print(df)
    return df
    
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_TSNE(out, y_true):
    tsne_np = TSNE(n_components = 2).fit_transform(out)

    tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])

    tsne_df['target'] = y_true

    tsne_df_0 = tsne_df[tsne_df['target'] == 0]
    tsne_df_1 = tsne_df[tsne_df['target'] == 1]

    area = 2**2

    plt.scatter(tsne_df_0['component 0'], tsne_df_0['component 1'], s = area, color = 'pink', label = 'setosa')
    plt.scatter(tsne_df_1['component 0'], tsne_df_1['component 1'], s = area, color = 'purple', label = 'versicolor')

    plt.xlabel('component 0')
    plt.ylabel('component 1')
    plt.legend()

    plt.savefig('boston.png')


def softmax(x):
    
    max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x


def load_data(dataset_train, dataset_dev, dataset_test_label, batch_size):
    
    TrainLoader = DataLoader(dataset_train, batch_size = batch_size)
    EvalLoader = DataLoader(dataset_dev, batch_size = batch_size)
    InferenceLoader = DataLoader(dataset_test_label, batch_size = batch_size)

    return TrainLoader, EvalLoader, InferenceLoader


def name_wandb(arg_name, config):
    
    pretrained = config['pretrained']
    
    if pretrained == "klue/roberta-large":
        pretrained = "Rl"
    elif pretrained == "monologg/koelectra-base-v3-discriminator":
        pretrained = "KoE"
    elif pretrained == "klue/roberta-base":
        pretrained = "Rb"
    elif pretrained == "kykim/electra-kor-base":
        pretrained = "kykE"
    elif pretrained == "beomi/KcELECTRA-base":
        pretrained = "KcE"
    elif pretrained == "beomi/KcELECTRA-base-v2022":
        pretrained = "KcE22"
    elif pretrained == "kykim/funnel-kor-base":
        pretrained = "fun"
    elif pretrained == "tunib/electra-ko-base":
        pretrained = "tunib"
        
    
    bs =  config["batch_size"]
    lr = config["lr"]
    seed = config["seed"]
    kFold, nSplit = config["K-Fold"].split('/')
    scheduler = config["scheduler"]
    
    if scheduler == "None":
        scheduler = "N"
    
    
    name = f"{arg_name}_{bs}_{lr}_{scheduler}_{kFold}F{nSplit}_rs{seed}_{pretrained}"
    
    return name