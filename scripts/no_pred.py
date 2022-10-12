from util.module_utils import *

label = ["aa.jsonl"]

test_data = jsonlload(label)

# 빈칸 뽑아내는 코드
aa = []

for i in range(len(test_data)):
    
    if test_data[i]['annotation'] == []:
        aa.append(test_data[i])
        
file_name = submissionPth + "ss"

jsondump(aa, f"{file_name}.jsonl")
    
exit()
# 빈칸 뽑아내는 코드 (end)