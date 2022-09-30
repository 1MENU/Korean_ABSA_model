import argparse
from scripts.baseline import evaluation_f1
from scripts.util.module_utils import jsonlload, jsonltoDataFrame
import numpy as np
import json
import pandas as pd

# TODO : 여기수정
from util.modl_utils import tsv_to_df, predPth, submissionPth

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser()

# 입력받을 인자값 등록
parser.add_argument('--name', default="defalut")
parser.add_argument('--task', required = True)
parser.add_argument('--preds', nargs='+')
parser.add_argument('--weights', type=float, nargs='+')
parser.add_argument('--label', required = True, nargs='+') # CD or SD

# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

task_name = args.task

pred_model = []
final_submission_pred = None
# TODO : CD에 대한 앙상블
# TODO : SD에 대한 앙상블
# 근데 평가하는 방법은 같으니까 하나의 함수로 가능하지 않을까

#예측값에 똑같은 가중치를 주어서 argmax하여 결과값도출
for i in range(len(args.preds)):
    pred = np.load(f'{predPth}{task_name}/{args.preds[i]}.npy')

    pred_model.append(pred)

    if final_submission_pred is None:
        final_submission_pred = pred_model[i] * args.weights[i]
    else:
        final_submission_pred += pred_model[i] * args.weights[i]
        
test_label_file_list = args.label  # test data
test_df = jsonltoDataFrame(test_label_file_list) #리스트 타입으로 변환

label= test_df["annotation"] # CD, SD둘다 annotation 사용
final_submission_pred = np.argmax(final_submission_pred, axis=1)

#결과랑 예측값 받아서 정확도 반환
result = evaluation_f1(true_data= label, pred_data=final_submission_pred)

print(f"[{task_name}] {args.name}")
print(args.preds)
print(args.weights)
print("f1 : ", result)


