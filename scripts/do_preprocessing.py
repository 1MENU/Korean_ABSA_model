
from copy import deepcopy
from base_data import *
from util.module_utils import *
from util.preprocessing import *

label = ["3Fold.jsonl"]
input_data = jsonlload(label)

for i in range(len(input_data)):
    
    form = input_data[i]['sentence_form']
    
    form = spacing_sent(form)
    
    # # 이모티콘 제거 
    # form = del_emoji_all(form)
    # # 반복제거
    # form = repeat_del(form, n=3)
    # # 텍스트 이모티콘 제거
    # form = remove_texticon(form)
    
    input_data[i]['sentence_form'] = form
    
    print(form)
    
file_name = datasetPth + "3Fold_spell"

jsondump(input_data, f"{file_name}.jsonl")