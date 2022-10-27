from CD_dataset import *
from CD_model import *
from CD_module import *

from SC_dataset import *
from SC_model import *
from SC_module import *

import math

parser = argparse.ArgumentParser()

parser.add_argument('--cd', required = True, nargs='+')
parser.add_argument('--sc', required = True, nargs='+')
parser.add_argument('--name', required = True)
parser.add_argument('-bs', '--batch_size', type=int, default=2048)

args = parser.parse_args()

device = torch.device("cuda")

CD_pretrained = "kykim/electra-kor-base"     # "beomi/KcELECTRA-base-v2022"
SC_pretrained = "kykim/electra-kor-base"


test_file_list = ["test.jsonl"]
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





def evaluation_f1_plus(true_data, pred_data):
    
    wrong = []
    
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
                
                if wrong==[]:
                    wrong.append(true_data_list[i])
                else:
                    if true_data_list[i] != wrong[-1]:
                        wrong.append(true_data_list[i])

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
                
                if wrong==[]:
                    wrong.append(true_data_list[i])
                else:
                    if true_data_list[i] != wrong[-1]:
                        wrong.append(true_data_list[i])


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

    return pipeline_result['F1'], wrong




label = ["data_new.jsonl"]
test_data = jsonlload(label)

score_1, wrong = evaluation_f1_plus(test_data, output_data)
print(score_1)

file_name = submissionPth + "wrong_answer_note_" + args.name
jsondump(wrong, f"{file_name}.json")