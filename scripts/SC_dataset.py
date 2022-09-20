from util.utils import *

## model BoolQ dataloader ##
class BoolQA_dataset(Dataset): 
    def __init__(self, df, pretrained_tokenizer): #str: path of csv
        super(BoolQA_dataset,self).__init__()

        self.data = df

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    
    def __len__(self): #return size of dataset(row*column)
        return len(self.data)
    
    def __name__(self): #return name
        return 'BoolQA_dataset'
        
    def __getitem__(self, item): #fetch data sample(row) for given key(item)
        qaData = self.data.iloc[item,:]
        max_text_length, max_question_length = TOKEN_MAX_LENGTH['BoolQ']
        #print(len(sentData), type(sentData), sentData) #4, Seires, (csv: source, accep_label, source_annot, sentence)

        text = qaData['Text'][:max_text_length] #passage
        question = qaData['Question'][:max_question_length] #question

        input_str = question+self.tokenizer.sep_token+text #[CLS:2]+question+[SEP:3]+text+[SEP]+[PAD:1]
        max_tokenizer_length = max_text_length+max_question_length+5
        tok = self.tokenizer(input_str, padding="max_length", max_length=max_tokenizer_length, truncation=True) #PreTrainedTokenizer.__call__(): str -> tensor(3*400)

        input_ids=torch.LongTensor(tok["input_ids"]) #input token index in vacabs
        token_type_ids=torch.LongTensor(tok["token_type_ids"]) #segment token index 
        attention_mask=torch.LongTensor(tok["attention_mask"]) #boolean: masking attention(0), not masked(1)

        label=qaData['Answer(FALSE = 0, TRUE = 1)']

        return input_ids, token_type_ids, attention_mask, label #tensor(400), tensor(400), tensor(400), int
        #return type(len): tensor(2*(textMaxlen+quesMaxlen)) x 3개, tensor(int)
        #current len: 485, 485, 485, 1


def tokenize_and_align_labels(tokenizer, form, annotations, max_len):

    global polarity_count
    global entity_property_count

    entity_property_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }
    polarity_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }

    for pair in entity_property_pair:
        isPairInOpinion = False
        if pd.isna(form):
            break
        tokenized_data = tokenizer(form, pair, padding='max_length', max_length=max_len, truncation=True)
        for annotation in annotations:
            entity_property = annotation[0]
            polarity = annotation[2]

            # # 데이터가 =로 시작하여 수식으로 인정된경우
            # if pd.isna(entity) or pd.isna(property):
            #     continue

            if polarity == '------------':
                continue


            if entity_property == pair:
                polarity_count += 1
                entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
                entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                entity_property_data_dict['label'].append(label_name_to_id['True'])

                polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
                polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                polarity_data_dict['label'].append(polarity_name_to_id[polarity])

                isPairInOpinion = True
                break

        if isPairInOpinion is False:
            entity_property_count += 1
            entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
            entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
            entity_property_data_dict['label'].append(label_name_to_id['False'])

    return entity_property_data_dict, polarity_data_dict


def get_dataset(raw_data, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []

    for utterance in raw_data:
        entity_property_data_dict, polarity_data_dict = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len)
        input_ids_list.extend(entity_property_data_dict['input_ids'])
        attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        token_labels_list.extend(entity_property_data_dict['label'])

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])

    print('polarity_data_count: ', polarity_count)
    print('entity_property_data_count: ', entity_property_count)

    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list),
                         torch.tensor(token_labels_list)), TensorDataset(torch.tensor(polarity_input_ids_list), torch.tensor(polarity_attention_mask_list),
                         torch.tensor(polarity_token_labels_list))