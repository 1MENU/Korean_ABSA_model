import argparse
from Doyeon.team1.scripts.util.utils import F1_scrore
from scripts.CD_dataset import get_CD_dataset
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
parser.add_argument('--task', required = True) # CD or SD
parser.add_argument('--preds', nargs='+')
parser.add_argument('--weights', type=float, nargs='+')
parser.add_argument('--label', required = True, nargs='+') # CD or SD

# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

task_name = args.task

pred_model = []
final_submission_pred = None
#예측값에 똑같은 가중치를 주어서 argmax하여 결과값도출
for i in range(len(args.preds)):
    pred = np.load(f'{predPth}{task_name}/{args.preds[i]}.npy')

    pred_model.append(pred)

    if final_submission_pred is None:
        final_submission_pred = pred_model[i] * args.weights[i]
    else:
        final_submission_pred += pred_model[i] * args.weights[i]
        
test_label_file_list = args.label  # test data의 정답값
test_data = jsonlload(test_label_file_list) #리스트 타입으로 변환
dataset_test , dataset_test , dataset_test = get_CD_dataset(test_data, test_data, test_data, args.pretrained)
# label= test_df["annotation"].to_list() # CD, SD둘다 annotation 사용
# label = sum(label, [])
# df = pd.DataFrame(label)

final_submission_pred = np.argmax(final_submission_pred, axis=1)

# 파일 어떻게 쓰는지 확인해야 수정가능
yy = np.logical_or(dataset_test, final_submission_pred)

y_true = dataset_test[np.where(np.logical_and(yy=1))]
y_pred = final_submission_pred[np.where(np.logical_and(yy=1))]

#결과랑 예측값 받아서 정확도 반환
result = F1_scrore(y_true, y_pred, average="binary")

print(f"[{task_name}] {args.name}")
print(args.preds)
print(args.weights)
print("f1 : ", result)


