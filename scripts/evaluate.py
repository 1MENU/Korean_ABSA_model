
from copy import deepcopy
from base_data import *
from util.module_utils import *

# parser = argparse.ArgumentParser()

# parser.add_argument('--pred', required = True)
# parser.add_argument('--test', required = True)

# args = parser.parse_args()

pred1 = ["1.json"]
pred2 = ["2.json"]
pred3 = ["3.json"]

label = ["lll.jsonl"]

pred_data1 = jsonlload(pred1)
pred_data2 = jsonlload(pred2)
pred_data3 = jsonlload(pred3)

test_data = jsonlload(label)

# for i in range(len(test_data)):
    
#     test_data[i]['annotation'] = []
    
#     for pair in entity_property_pair:
#         y_category = pair
#         aa = ["null"]
#         y_polarity = "positive"
        
#         test_data[i]['annotation'].append([y_category, aa, y_polarity])
        
# test_data[0]['annotation'] = []

score_1 = evaluation_f1(test_data, pred_data1)
score_1 = abs(0.5899 - score_1['entire pipeline result']['F1']) * 100
print("score_1 : ", score_1)

score_2 = evaluation_f1(test_data, pred_data2)
score_2 = abs(0.5623 - score_2['entire pipeline result']['F1']) * 100
print("score_2 : ", score_2)

# score_3 = evaluation_f1(test_data, pred_data3)
# score_3 = abs(0.0008 - score_3['entire pipeline result']['F1']) * 100
# print("score_3 : ", score_3)

save_1 = score_1
save_2 = score_2

first_1 = score_1
first_2 = score_2

copy_data = deepcopy(test_data)

exit()

for i in range(len(test_data)):
    
    test_data[i]['annotation'] = []
    
    for pair in entity_property_pair:
        y_category = pair
        aa = ["null"]
        
        for polarity in polarity_id_to_name:
            
            y_polarity = polarity
            
            test_data[i]['annotation'].append([y_category, aa, y_polarity])
            
            
            score_1 = evaluation_f1(test_data, pred_data1)
            score_1 = abs(0.5899 - score_1['entire pipeline result']['F1']) * 100
            # print("score_1 : ", score_1)

            score_2 = evaluation_f1(test_data, pred_data2)
            score_2 = abs(0.5623 - score_2['entire pipeline result']['F1']) * 100
            # print("score_2 : ", score_2)
            
            test_data[i]['annotation'].remove([y_category, aa, y_polarity])
            
            if save_1 <= score_1 or save_2 <= score_2 :
                pass
            
            else:
                save_1 = score_1
                save_2 = score_2
                
                test_data[i]['annotation'].append([y_category, aa, y_polarity])
                
                copy_data[i]['annotation'].append([y_category, aa, y_polarity])
                
                print(copy_data[i])
                break
            
    test_data[i]['annotation'] = []
    save_1 = first_1
    save_2 = first_2
        
    
# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        # json.dump(j, f, ensure_ascii=False)
        for i in j:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

file_name = submissionPth + "lll"

jsondump(copy_data, f"{file_name}.jsonl")

# 860 ~ 1289