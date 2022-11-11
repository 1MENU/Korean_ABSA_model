from CD_dataset import *
from CD_model import *
from CD_module import *

from SC_dataset import *
from SC_model import *
from SC_module import *

import math

parser = argparse.ArgumentParser()

parser.add_argument('--test_file', nargs='+')
parser.add_argument('--cd', required = True, nargs='+')
parser.add_argument('--sc', required = True, nargs='+')
parser.add_argument('--name', required = True)
parser.add_argument('-bs', '--batch_size', type=int, default=2048)
parser.add_argument('--pretrained', default="kykim/electra-kor-base")

args = parser.parse_args()

device = torch.device("cuda")

CD_pretrained = args.pretrained
SC_pretrained = args.pretrained

test_file_list = args.test_file
test_data = jsonlload(test_file_list)

_, _, dataset_test = get_CD_dataset(test_data, test_data, test_data, CD_pretrained, max_len = 90)

_, _, InferenceLoader = load_data(dataset_test, dataset_test, dataset_test, batch_size = args.batch_size)

CD_preds = []

for i in range(len(args.cd)):
    CD_pred_model = CD_model(CD_pretrained)
    CD_pred_model.load_state_dict(torch.load(f'{saveDirPth_str}CD/{args.cd[i]}.pt'))
    CD_pred_model.to(device)
    
    model_pred = CD_inference_model(CD_pred_model, InferenceLoader, device)
    
    print(model_pred)
    print("fin")
    
    CD_preds.append(model_pred)


pred_model = []
final_submission_pred = None
#예측값에 똑같은 가중치를 주어서 argmax하여 결과값도출
for p in range(len(CD_preds)):

    pred_model.append(CD_preds[p])

    if final_submission_pred is None:
        final_submission_pred = pred_model[p]
    else:
        final_submission_pred += pred_model[p]

final_pred = np.argmax(final_submission_pred, axis=1)

one_list = list(filter(lambda x: final_pred[x] == 1, range(len(final_pred))))  # 1인 index들을 return


assert len(final_pred)/len(entity_property_pair) == len(test_data)

SC_list = []

for i in one_list:
    sent_i = math.floor(i / len(entity_property_pair))
    pair_i = i % len(entity_property_pair)
    
    SC_list.append((sent_i, pair_i))
    
    print(test_data[sent_i]['sentence_form'], entity_property_pair[pair_i])


## SC
SC_test_data = [dataset_test[i] for i in one_list]

_, _, InferenceLoader = load_data(SC_test_data, SC_test_data, SC_test_data, batch_size = args.batch_size)

SC_preds = []

for i in range(len(args.sc)):
    SC_pred_model = SC_model(SC_pretrained)
    SC_pred_model.load_state_dict(torch.load(f'{saveDirPth_str}SC/{args.sc[i]}.pt'))
    SC_pred_model.to(device)
    
    model_pred = SC_inference_model(SC_pred_model, InferenceLoader, device)
    
    print(model_pred)
    print("fin")
    
    SC_preds.append(model_pred)
    
    
    

pred_model = []
final_submission_pred = None
#예측값에 똑같은 가중치를 주어서 argmax하여 결과값도출
for p in range(len(SC_preds)):

    pred_model.append(SC_preds[p])

    if final_submission_pred is None:
        final_submission_pred = pred_model[p]
    else:
        final_submission_pred += pred_model[p]

final_pred = np.argmax(final_submission_pred, axis=1)




print(len(final_pred))
print(len(one_list))

assert len(final_pred) == len(one_list)

output_data = copy.deepcopy(test_data)

for i in range(len(final_pred)):
    index, pair_i = SC_list[i]
    polarity = final_pred[i]
    
    output_data[index]['annotation'].append([entity_property_pair[pair_i], polarity_id_to_name[polarity]])


file_name = submissionPth + args.name

jsondump(output_data, f"{file_name}.json")