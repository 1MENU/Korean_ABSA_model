
from copy import deepcopy
from base_data import *
from util.module_utils import *




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