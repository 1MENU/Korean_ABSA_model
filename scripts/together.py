from CD_dataset import *
from CD_model import *
from CD_module import *

from SC_dataset import *
from SC_model import *
from SC_module import *

parser = argparse.ArgumentParser()

parser.add_argument('--cd', required = True)
parser.add_argument('--sc', required = True)
parser.add_argument('--name', required = True)
parser.add_argument('-bs', '--batch_size', type=int, default=128)

args = parser.parse_args()

device = torch.device("cuda")


test_file_list = ["test.jsonl"]
test_data = jsonlload(test_file_list)

CD_pretrained = "kykim/electra-kor-base"     # "kykim/electra-kor-base"    # "kykim/funnel-kor-base"
SC_pretrained = "beomi/KcELECTRA-base"


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

print(output_data)

for sentence in output_data:
    
    form = sentence['sentence_form']
    sentence['annotation'] = []
    
    if type(form) != str:
        print("form type is arong: ", form)
        continue
    
    # form_spell = spacing_sent(form)
    
    for pair in entity_property_pair:
        
        # 이 자리에 전처리
        
        final_pair = pair
        # final_pair = replace_htag(final_pair)
        
        # sent = pair + CD_tokenizer.cls_token + form_spells
        
        tokenized_data = CD_tokenizer(final_pair, form, padding='max_length', max_length=256, truncation=True)

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