from util.utils import *
from util.preprocessing import *
from base_data import *

# import logging
# # 로그 생성
# logger = logging.getLogger()
# # 로그의 출력 기준 설정
# logger.setLevel(logging.INFO)
# # log 출력 형식
# formatter = logging.Formatter('%(message)s')
# # log 출력
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)
# # log를 파일에 출력
# file_handler = logging.FileHandler('my.log')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


#         # 밑에 넣을 코드
#         tokenized_text = tokenizer.tokenize(form)
#         logger.info(form)
#         logger.info(tokenized_text)



def CD_dataset(raw_data, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    token_labels_list = []
    
    e1_mask_list = []
    e2_mask_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_type_ids_list = []
    polarity_token_labels_list = []

    for utterance in raw_data:
        
        # 이 자리에 전처리 가능
        
        form = utterance['sentence_form']
        
        form = my_preprocessing(form)
        
        # print(form)
        

        entity_property_data_dict, polarity_data_dict = tokenize_and_align_labels(tokenizer, form, utterance['annotation'], max_len)
        
        input_ids_list.extend(entity_property_data_dict['input_ids'])
        token_type_ids_list.extend(entity_property_data_dict['token_type_ids'])
        attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        token_labels_list.extend(entity_property_data_dict['label'])
        
        e1_mask_list.extend(entity_property_data_dict['e1_mask'])
        e2_mask_list.extend(entity_property_data_dict['e2_mask'])

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_token_type_ids_list.extend(polarity_data_dict['token_type_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])


    return TensorDataset(
        torch.tensor(input_ids_list),
        torch.tensor(token_type_ids_list),
        torch.tensor(attention_mask_list),
        torch.tensor(token_labels_list),
        torch.tensor(e1_mask_list),
        torch.tensor(e2_mask_list)
        ), TensorDataset(
            torch.tensor(polarity_input_ids_list), 
            torch.tensor(polarity_token_type_ids_list),
            torch.tensor(polarity_attention_mask_list),
            torch.tensor(polarity_token_labels_list)
        )


def tokenize_and_align_labels(tokenizer, form, annotations, max_len):

    entity_property_data_dict = {
        'input_ids': [],
        'token_type_ids' : [],
        'attention_mask': [],
        'label': [],
        'e1_mask' : [],
        'e2_mask' : []
    }
    polarity_data_dict = {
        'input_ids': [],
        'token_type_ids' : [],
        'attention_mask': [],
        'label': []
    }

    for pair in entity_property_pair:
        isPairInOpinion = False
        if pd.isna(form):
            break
        
        # 이 자리에는 toknizer에 들어갈 구조 변경 가능
        
        pair_final = pair
        
        tokenized_data = tokenizer(form, pair_final, padding='max_length', max_length=max_len, truncation=True)
        
        for annotation in annotations:
            entity_property = annotation[0]
            polarity = annotation[2]

            # # 데이터가 =로 시작하여 수식으로 인정된경우
            # if pd.isna(entity) or pd.isna(property):
            #     continue

            if polarity == '------------':
                continue


            if entity_property == pair:
                entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
                entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                entity_property_data_dict['token_type_ids'].append(tokenized_data['token_type_ids'])
                entity_property_data_dict['label'].append(label_name_to_id['True'])

                polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
                polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                polarity_data_dict['token_type_ids'].append(tokenized_data['token_type_ids'])
                polarity_data_dict['label'].append(polarity_name_to_id[polarity])

                isPairInOpinion = True
                break

        if isPairInOpinion is False:
            entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
            entity_property_data_dict['token_type_ids'].append(tokenized_data['token_type_ids'])
            entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
            entity_property_data_dict['label'].append(label_name_to_id['False'])

        
        first_sep = tokenized_data['input_ids'].index(3)
        last_sep = tokenized_data['input_ids'][first_sep+1:].index(3)
        
        e1_mask = [0] * len(tokenized_data['input_ids'])
        e2_mask = [0] * len(tokenized_data['input_ids'])
        
        # # 여기가 뽑아낼 부분
        # # 지금은 2번째 CLS만 뽑아보자
        # e2_mask[second_cls] = 1
        
        # # 지금은 e1, e2 다 뽑아보자
        # for i in range(1, second_cls):
        #     e1_mask[i] = 1
        # for i in range(second_cls + 1, last_sep):
        #     e2_mask[i] = 1
        
        for i in range(first_sep + 1, first_sep + 1 + last_sep):
            e2_mask[i] = 1
            
        entity_property_data_dict['e1_mask'].append(e1_mask)
        entity_property_data_dict['e2_mask'].append(e2_mask)
        
    return entity_property_data_dict, polarity_data_dict


def get_CD_dataset(train_data, dev_data, test_data, pretrained_tokenizer, max_len):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    
    train_CD_data, train_SC_data = CD_dataset(train_data, tokenizer, max_len)
    dev_CD_data, dev_SC_data = CD_dataset(dev_data, tokenizer, max_len)
    test_CD_data, test_SC_data = CD_dataset(test_data, tokenizer, max_len)

    return train_CD_data, dev_CD_data, test_CD_data

