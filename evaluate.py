import argparse
from scripts.util.module_utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--pred', required = True)
parser.add_argument('--test', required = True)

args = parser.parse_args()

pred_data = jsonlload(args.pred)

test_data = jsonlload(args.label)

f1 = evaluation_f1(test_data, pred_data)

print(f1)