import argparse
from util.module_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--pred', required = True)
parser.add_argument('--test', required = True)

args = parser.parse_args()

pred = [args.pred]
label = [args.test]

pred_data = jsonlload(pred)

test_data = jsonlload(label)

f1 = evaluation_f1(test_data, pred_data)

print("F1 score : ", f1['entire pipeline result']['F1'] * 100)