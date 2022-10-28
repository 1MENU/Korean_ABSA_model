from base_data import *
from util.module_utils import *
import json

data_list = [] # 문장들 저장 
data_sent=[]
data_annotation=[]
label_list=[]
data_id=[]


test = ["test.jsonl"]
test_data = jsonlload(test)

our = ["data_new.jsonl"]
our_data = jsonlload(our)


assert len(test_data) == len(our_data)
      
for i in range(len(test_data)):
  data_sent.append(test_data[i]['sentence_form'])
  data_id.append(test_data[i]["id"])
  
for i in range(len(our_data)):
  annot = our_data[i]["annotation"]
  test_data[i]["annotation"]=annot


file_name = submissionPth + "data_new_update"
jsondump(test_data, f"{file_name}.jsonl")
