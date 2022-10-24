
from copy import deepcopy
from base_data import *
from util.module_utils import *

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




# our_pred = ["../materials/submission/ttttt.json"]
# our_pred_data = jsonlload(our_pred)

# label = ["data_new.jsonl"]
# test_data = jsonlload(label)

# score_1, wrong = evaluation_f1_plus(test_data, our_pred_data)
# print(score_1)

# file_name = submissionPth + "wrong"
# jsondump(wrong, f"{file_name}.json")

# exit()








pred1 = ["59_98.json"]
pred2 = ["62_14.json"]
pred3 = ["1.json"]
pred4 = ["4.json"]
pred3 = ["5.json"]
pred4 = ["6.json"]

pred_data1 = jsonlload(pred1)
pred_data2 = jsonlload(pred2)
pred_data3 = jsonlload(pred3)
pred_data4 = jsonlload(pred4)
pred_data5 = jsonlload(pred3)
pred_data6 = jsonlload(pred4)

label = ["data_new.jsonl"]
test_data = jsonlload(label)



score_1 = evaluation_f1(test_data, pred_data1)
score_1 = abs(0.5998 - score_1['entire pipeline result']['F1']) * 100
print("score_1 : ", score_1)

score_2 = evaluation_f1(test_data, pred_data2)
score_2 = abs(0.6214 - score_2['entire pipeline result']['F1']) * 100
print("score_2 : ", score_2)

score_3 = evaluation_f1(test_data, pred_data3)
score_3 = abs(0.5899 - score_3['entire pipeline result']['F1']) * 100
print("score_3 : ", score_3)

score_4 = evaluation_f1(test_data, pred_data4)
score_4 = abs(0.5786 - score_4['entire pipeline result']['F1']) * 100
print("score_4 : ", score_4)

score_5 = evaluation_f1(test_data, pred_data5)
score_5 = abs(0.5549 - score_5['entire pipeline result']['F1']) * 100
print("score_5 : ", score_5)

score_6 = evaluation_f1(test_data, pred_data6)
score_6 = abs(0.5831 - score_6['entire pipeline result']['F1']) * 100
print("score_6 : ", score_6)


save_1 = score_1
save_2 = score_2
save_3 = score_3
save_4 = score_4
save_5 = score_5
save_6 = score_6