from util.module_utils import *

from CD_module import *
from base_data import *


test_file_list = ["62_14.json"]


data = jsonlload(test_file_list)

copy = copy.deepcopy(data)

for s in range(len(copy)):

    data[s]['annotation'] = []

    for y_ano in range(len(copy[s]['annotation'])):

        y_category = copy[s]['annotation'][y_ano][0]
        aa = ["null"]
        y_polarity = copy[s]['annotation'][y_ano][1]

        data[s]['annotation'].append([y_category, aa, y_polarity])

print(data)

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        # json.dump(j, f, ensure_ascii=False)
        for i in j:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

file_name = submissionPth + "data"

jsondump(data, f"{file_name}.jsonl")

# label = ["lll.jsonl"]

# test_data = jsonlload(label)

# # 빈칸 뽑아내는 코드
# aa = []

# for i in range(len(test_data)):
    
#     if test_data[i]['annotation'] == []:
#         aa.append(test_data[i])
        
# file_name = submissionPth + "tt"

# jsondump(aa, f"{file_name}.jsonl")
    
# exit()
# # 빈칸 뽑아내는 코드 (end)