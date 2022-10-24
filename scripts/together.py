from CD_dataset import *
from CD_model import *
from CD_module import *

from SC_dataset import *
from SC_model import *
from SC_module import *

import math

parser = argparse.ArgumentParser()

parser.add_argument('--cd', required = True, nargs='+')
parser.add_argument('--sc', required = True, nargs='+')
parser.add_argument('--name', required = True)
parser.add_argument('-bs', '--batch_size', type=int, default=256)

args = parser.parse_args()

device = torch.device("cuda")

CD_pretrained = "kykim/electra-kor-base"     # "kykim/electra-kor-base"
SC_pretrained = "kykim/electra-kor-base"


test_file_list = ["test.jsonl"]
test_data = jsonlload(test_file_list)

_, _, dataset_test = get_CD_dataset(test_data, test_data, test_data, CD_pretrained, max_len = 90)

_, _, InferenceLoader = load_data(dataset_test, dataset_test, dataset_test, batch_size = args.batch_size)

CD_preds = []

for i in range(len(args.cd)):
    CD_pred_model = CD_model(CD_pretrained)
    CD_pred_model.load_state_dict(torch.load(f'{saveDirPth_str}CD/{args.cd[i]}.pt'))
    CD_pred_model.to(device)
    
    model_pred = CD_inference_model(CD_pred_model, InferenceLoader, device)
    
    print(model_pred)
    print("fin")
    
    CD_preds.append(model_pred)


pred_model = []
final_submission_pred = None
#예측값에 똑같은 가중치를 주어서 argmax하여 결과값도출
for p in range(len(CD_preds)):

    pred_model.append(CD_preds[p])

    if final_submission_pred is None:
        final_submission_pred = pred_model[p]
    else:
        final_submission_pred += pred_model[p]

final_pred = np.argmax(final_submission_pred, axis=1)

one_list = list(filter(lambda x: final_pred[x] == 1, range(len(final_pred))))  # 1인 index들을 return


assert len(final_pred)/len(entity_property_pair) == len(test_data)

SC_list = []

for i in one_list:
    sent_i = math.floor(i / len(entity_property_pair))
    pair_i = i % len(entity_property_pair)
    
    SC_list.append((sent_i, pair_i))
    
    print(test_data[sent_i]['sentence_form'], entity_property_pair[pair_i])


## SC
SC_test_data = [dataset_test[i] for i in one_list]

_, _, InferenceLoader = load_data(SC_test_data, SC_test_data, SC_test_data, batch_size = args.batch_size)

SC_preds = []

for i in range(len(args.sc)):
    SC_pred_model = SC_model(SC_pretrained)
    SC_pred_model.load_state_dict(torch.load(f'{saveDirPth_str}SC/{args.sc[i]}.pt'))
    SC_pred_model.to(device)
    
    model_pred = SC_inference_model(SC_pred_model, InferenceLoader, device)
    
    print(model_pred)
    print("fin")
    
    SC_preds.append(model_pred)
    
    
    

pred_model = []
final_submission_pred = None
#예측값에 똑같은 가중치를 주어서 argmax하여 결과값도출
for p in range(len(SC_preds)):

    pred_model.append(SC_preds[p])

    if final_submission_pred is None:
        final_submission_pred = pred_model[p]
    else:
        final_submission_pred += pred_model[p]

final_pred = np.argmax(final_submission_pred, axis=1)




print(len(final_pred))
print(len(one_list))

assert len(final_pred) == len(one_list)

output_data = copy.deepcopy(test_data)

for i in range(len(final_pred)):
    index, pair_i = SC_list[i]
    polarity = final_pred[i]
    
    output_data[index]['annotation'].append([entity_property_pair[pair_i], polarity_id_to_name[polarity]])


file_name = submissionPth + args.name

jsondump(output_data, f"{file_name}.json")


exit()





exit()


CD_model = CD_model(CD_pretrained)
CD_model.load_state_dict(torch.load(f'{saveDirPth_str}CD/{args.cd}.pt'))
CD_model.to(device)

SC_model = SC_model(SC_pretrained)
SC_model.load_state_dict(torch.load(f'{saveDirPth_str}SC/{args.sc}.pt'))
SC_model.to(device)

CD_model.eval()
SC_model.eval()



CD_tokenizer = AutoTokenizer.from_pretrained(CD_pretrained)
num_added_toks = CD_tokenizer.add_special_tokens(special_tokens_dict)
    
SC_tokenizer = AutoTokenizer.from_pretrained(SC_pretrained)
num_added_toks = SC_tokenizer.add_special_tokens(special_tokens_dict)










output_data = copy.deepcopy(test_data)


for sentence in output_data:
    
    form = sentence['sentence_form']
    sentence['annotation'] = []
    
    if type(form) != str:
        print("form type is arong: ", form)
        continue
    
    # form_spell = spacing_sent(form)
    
    for pair in entity_property_pair:
        
        # 이 자리에 전처리
        
        pair_final = pair
        
        # final_pair = replace_htag(final_pair)
        
        sent = form + CD_tokenizer.cls_token + pair_final
        
        tokenized_data = CD_tokenizer(sent, padding='max_length', max_length=100, truncation=True)

        input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
        token_type_ids = torch.tensor([tokenized_data['token_type_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
        
        with torch.no_grad():
            output = CD_model(input_ids, token_type_ids, attention_mask)

        ce_predictions = torch.argmax(output, dim = -1)

        if ce_predictions == 1:
            
            # 이 자리에 전처리
        
            tokenized_data = SC_tokenizer(form, pair, padding='max_length', max_length=256, truncation=True)
            
            input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
            token_type_ids = torch.tensor([tokenized_data['token_type_ids']]).to(device)
            attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
            
            with torch.no_grad():
                output = SC_model(input_ids, token_type_ids, attention_mask)

            SC_predictions = torch.argmax(output, dim=-1)
            
            if SC_predictions == 0: SC_result = 'positive'
            elif SC_predictions == 1: SC_result = 'negative'
            elif SC_predictions == 2: SC_result = 'neutral'
            

            sentence['annotation'].append([pair, SC_result])


# output_data가 결과물

# print('F1 result: ', evaluation_f1(test_data, output_data))

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        # json.dump(j, f, ensure_ascii=False)
        for i in j:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")

file_name = submissionPth + args.name

jsondump(output_data, f"{file_name}.json")