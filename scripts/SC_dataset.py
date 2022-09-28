from util.utils import *
from base_data import *

def SC_dataset(raw_data, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    token_labels_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_type_ids_list = []
    polarity_token_labels_list = []

    for utterance in raw_data:

        entity_property_data_dict, polarity_data_dict = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len)
        
        input_ids_list.extend(entity_property_data_dict['input_ids'])
        token_type_ids_list.extend(entity_property_data_dict['token_type_ids'])
        attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        token_labels_list.extend(entity_property_data_dict['label'])

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_token_type_ids_list.extend(polarity_data_dict['token_type_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])

    return TensorDataset(
        torch.tensor(input_ids_list),
        torch.tensor(token_type_ids_list),
        torch.tensor(attention_mask_list),
        torch.tensor(token_labels_list)
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
        'label': []
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
        
        # 이 자리에 전처리 할 수 있음. form 변형
        # def preprocessing(form) return str
        # tokenizer에 어떻게 들어갈지 형식도 변형
        
        tokenized_data = tokenizer(pair, form, padding='max_length', max_length=max_len, truncation=True)
        
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

    return entity_property_data_dict, polarity_data_dict


def get_CD_dataset(train_data, dev_data, test_data, pretrained_tokenizer):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    train_CD_data, train_SC_data = SC_dataset(train_data, tokenizer, 256)
    dev_CD_data, dev_SC_data = SC_dataset(dev_data, tokenizer, 256)
    test_CD_data, test_SC_data = SC_dataset(test_data, tokenizer, 256)

    return train_SC_data, dev_SC_data, test_SC_data