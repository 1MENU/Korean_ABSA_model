from CD_module import *
from base_data import *

def custom_stratified_KFold(file_list, n_splits, which_k):
    
    def jsonlload(fname_list, encoding="utf-8"):
        json_list = []

        for index, value in enumerate(fname_list):
            fname = "../dataset/" + value

            with open(fname, encoding=encoding) as f:
                for line in f.readlines():
                    # print(line)
                    json_list.append(json.loads(line))

        return json_list

    data = jsonlload(file_list)
    label = []
    
    for d in data:
        
        annotation = d["annotation"]
        
        ano_index = entity_property_pair.index(annotation[0][0])
        
        for a in annotation:
            if entity_property_pair.index(a[0]) > ano_index:
                ano_index = entity_property_pair.index(a[0])
        
        label.append(ano_index)


    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=1)
    
    n_iter = 0

    for train_idx, test_idx in skf.split(data, label):
        n_iter += 1

        # print(test_idx)

        if n_iter == which_k:
            print(f'------------------ {n_splits}-Fold 중 {n_iter}번째 ------------------')
            
            features_train = [data[i] for i in train_idx]
            features_test = [data[i] for i in test_idx]
            
            file_name = '../dataset/' + str(n_iter) + "Fold"

            jsondump(features_test, f"{file_name}.jsonl")

            return features_train, features_test

device = torch.device('cuda')
set_seed(1, device) #random seed 정수로 고정.

input_file_list = ["train.jsonl", "dev.jsonl", "temp_aug.jsonl"]


train_data, dev_data = custom_stratified_KFold(input_file_list, 3, 1)

train_data, dev_data = custom_stratified_KFold(input_file_list, 3, 2)

train_data, dev_data = custom_stratified_KFold(input_file_list, 3, 3)