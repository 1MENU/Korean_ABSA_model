from util.utils import *
from base_data import *

def CD_dataset(raw_data, tokenizer, max_len):
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

        # 이 자리에 전처리 가능
        
        # form = replace_marks(form)
        
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

    train_CD_data, train_SC_data = CD_dataset(train_data, tokenizer, 256)
    dev_CD_data, dev_SC_data = CD_dataset(dev_data, tokenizer, 256)
    test_CD_data, test_SC_data = CD_dataset(test_data, tokenizer, 256)

    return train_CD_data, dev_CD_data, test_CD_data

"""
1. 이모티콘 → 텍스트 대체
    - Test set 에는 이모티콘 개수 자체는 적음
2. 텍스트 이모지 (ex, :) , ^^, >.<, > 3 <, // _ //, ㅋ.ㅋ, (--)(__) )
    1. 이모티콘
    2. ㅎㅎㅎ 혹은 ㅋㅋㅋ
    
3. ~~데이터 증강~~
4. 음성어, 의태어 처리 
    1. 토크나이저를 돌려보았을 때의 결과 (아래 참고)
    2. 5번, 6번 포함
5. 정규화(normalization) : ㅋㅋㅋㅋㅋㅋㅋㅋㅋ, ㅎㅎㅎㅎㅎㅎㅎㅎㅎㅎ, ㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠㅠ,  ;;;;;;;;;; 같은거 
tokenize 되는 사이즈에 맞게 통일하기. (그냥이랑 / ## 붙어서 나오는 버전이랑 나눠서 ex) ㅎㅎㅎㅎ, #ㅎㅎ) (근데 이거 toknizer에 따라 다름) → KCELECTRA 에서 해결 가능
6. 반복되는 어미 ( “강추추추추추…”)
7. # (해시테그), 삭제
8. 띄어쓰기, 맞춤법 교정기 api 써서 전처리
    1. ex) “부드러운 잴느낌이랄까”
    
    https://github.com/Beomi/KcELECTRA
    
"""

def replace_marks(sentence):
    """
    tokenization 전
    strip() 양 끝의 대상 제거
    re.sub() 대상 변경. 공백이면 삭제
    sentence = re.sub('', '', sentence)
    """
    print(sentence)
    # 텍스트 이모지
    sentence = re.sub('\^\^', 'ㅋㅋ', sentence)
    sentence = re.sub(':\)', 'ㅋㅋ', sentence)
    sentence = re.sub('>.<', 'ㅋㅋ', sentence)
    sentence = re.sub('> 3 <', 'ㅋㅋ', sentence)
    sentence = re.sub('// _ //', 'ㅋㅋ', sentence)
    sentence = re.sub('ㅋ.ㅋ', 'ㅋㅋ', sentence)
    sentence = re.sub('\(--\)\(__\)', 'ㅋㅋ', sentence)
    sentence = re.sub('💝', '❤', sentence)
    sentence = re.sub('ㅠㅅㅜ', 'ㅠㅠ', sentence)
    sentence = re.sub(':D', 'ㅎㅎ', sentence)
    sentence = re.sub('\+_\+/', 'ㅋㅋ', sentence)
    sentence = re.sub('\^-\^*', 'ㅎㅎ', sentence)
    sentence = re.sub('ㅎ_ㅎ', 'ㅎㅎ', sentence)
    sentence = re.sub('-_-', '', sentence)
    # sentence = re.sub('', '', sentence)
        
    # 해시태그 바꾸기
    sentence = re.sub('#', ', ', sentence)
    

    # 👍🏻 👌 🤡👠 🎵 🍰🎂 🙋🏻 🙏🏻 𖤐➰ 🌹💋😲🖒💆‍♀😡👌 😴💧🙆‍♂ 😺🙆‍♂💆🏻‍♀🙆🏻🌻😮🐥🌝 \\ dev데이터셋
    # 🌹 👦🏼 👍🏻👏🏻🤘💡🍼 😲🙃🐱 🕺💝🕷🕸🏃‍♀✌🏻 💋💄📸💯💋👌🚗💬 🤮🎵🍎➰ 👆💎 🍷😜 🙆‍♂🖐💧🙋🏻‍♀ // train
    # 이모지 통일/감소
    sentence = re.sub('👍🏻', '👍', sentence)
    sentence = re.sub('😄', '👍 ', sentence)
    sentence = re.sub('🖒', '👍', sentence)
    sentence = re.sub('👌', '👍', sentence)
    sentence = re.sub('🤡', '👍', sentence)
    sentence = re.sub('👠', '👍', sentence)
    sentence = re.sub('🎵', '👍', sentence)
    sentence = re.sub('🍰', '👍', sentence)
    sentence = re.sub('🎂', '👍', sentence)
    sentence = re.sub('🙋🏻', '👍', sentence)
    sentence = re.sub('🙏🏻', '👍', sentence)
    sentence = re.sub('𖤐', '👍', sentence)
    sentence = re.sub('➰', '👍', sentence)
    sentence = re.sub('🌹', '👍', sentence)
    sentence = re.sub('💋', '👍', sentence)
    sentence = re.sub('😲', '👍', sentence)
    sentence = re.sub('💆‍♀', '👍', sentence)
    sentence = re.sub('😡', '👍', sentence)
    sentence = re.sub('😴', '👍', sentence)
    sentence = re.sub('💧', '👍', sentence)
    sentence = re.sub('🙆‍♂', '👍', sentence)
    sentence = re.sub('😺', '👍', sentence)
    sentence = re.sub('💆🏻‍♀', '👍', sentence)
    sentence = re.sub('🙆🏻', '👍', sentence)
    sentence = re.sub('🌻', '👍', sentence)
    sentence = re.sub('😮', '👍', sentence)
    sentence = re.sub('🐥', '👍', sentence)
    sentence = re.sub('🌝', '👍', sentence)
    sentence = re.sub('👦🏼', '👍', sentence)
    sentence = re.sub('👏🏻', '👍', sentence)
    sentence = re.sub('🤘', '👍', sentence)
    sentence = re.sub('💡', '👍', sentence)
    sentence = re.sub('🍼', '👍', sentence)
    sentence = re.sub('😲', '👍', sentence)
    sentence = re.sub('🙃', '👍', sentence)
    sentence = re.sub('🐱', '👍', sentence)
    sentence = re.sub('🕺', '👍', sentence)
    sentence = re.sub('🕷', '싫다 ', sentence)
    sentence = re.sub('🕸', '싫다 ', sentence)
    sentence = re.sub('🏃‍♀', '👍', sentence)
    sentence = re.sub('✌🏻', '👍', sentence)
    sentence = re.sub('💯', '👍', sentence)
    sentence = re.sub('🤮', '싫다 ', sentence)
    sentence = re.sub('😜', '👍', sentence)
    sentence = re.sub('🖐', '👍', sentence)
    
    # 반복 문자 삭제 ㅋㅋㅋㅋㅋㅋ, ㅎㅎㅎㅎㅎㅎ, 강추추추추추
    sentence = repeat_normalize(sentence, num_repeats=2)
       
    return sentence

def replace_htag(annotation):
    # 해시태그 바꾸기
    annotation = re.sub('#', ', ', annotation)
    return annotation
