import pandas as pd
import json

# jsonl 파일 읽어서 list에 저장
def jsonlload(fname_list, encoding="utf-8"):
    json_list = []

    for index, value in enumerate(fname_list):
        fname = "dataset/" + value

        with open(fname, encoding=encoding) as f:
            for line in f.readlines():
                json_list.append(json.loads(line))

    return json_list

test_label_file_list = ["train.jsonl"]
test_data = jsonlload(test_label_file_list)

data = []

for utterance in test_data:
    data += [str(utterance['sentence_form'])]
    
    print(utterance['sentence_form'])


print(data)

df = pd.DataFrame(data)
writer = pd.ExcelWriter('test.xlsx')
df.to_excel(writer, sheet_name='welcome', index=False)
writer.save()


