from util.utils import *
from base_data import *
import re
from soynlp.normalizer import *
from hanspell import spell_checker
from bs4 import BeautifulSoup as bf


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

        # 이 자리에 전처리 할 수 있음. utterance['sentence_form'] 변형
        # def preprocessing(utterance['sentence_form']) return str
        
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
        
        form=replace_marks(form)
        #print(type(form))
        #input()
        sent = pair + tokenizer.cls_token + form
        
        tokenized_data = tokenizer(sent, padding='max_length', max_length=max_len, truncation=True)
        
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
수정필요 
def spacing_sent(sentence):
    #Py-Hanspell 이용 - Py-Hanspell은 네이버 한글 맞춤법 검사기를 바탕으로 만들어진 패키지
    #sentence=replace_htag(sentence)
    sentence="#&name&이유식 들어가면서 만들어줘야겠다.. 생각했는데 역시나 넘나 간편한 #베이비쿡솔로 .."
    result_train = spell_checker.check(sentence)
    sentence = result_train.as_dict()['checked']
    sentence = bf(sentence, features="html.parser")
   # print("change")
    print(sentence)
    #input()
    return sentence 
"""

def replace_marks(sentence):
    """
    tokenization 전 통일하지 않은 문장부호 제거하는 함수
    strip() 양 끝의 대상 제거
    re.sub() 대상 변경. 공백이면 삭제
    """
    print(sentence)
    # 💝이거는 ❤
    # 긍정, 중립, 부정 이모티콘 하나로 통일?
    # ㅠㅅㅜ :D ^^ +_+/ ^-^* ㅎ_ㅎ
    # 어 근데 tokenizer에서 ^^ 어떻게 처리되나 확인 먼저
    # ㅠㅠㅠㅠㅠㅠㅠㅠㅠ 나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 두 개로 바꾸기...... 어케한담.........
    # ㅠㅜㅠ ㅜㅠㅜ ㅜㅜ 통일? 좀 어려울듯
    # ;;;;;;; 이런거
    # /^[ㄱ-ㅎ|가-힣]+$/ 한글 1개 이상

    # 텍스트 이모지
    sentence = re.sub('\^\^', '', sentence)
    sentence = re.sub(':\)', '', sentence)
    sentence = re.sub('>.<', '', sentence)
    sentence = re.sub('> 3 <', '', sentence)
    sentence = re.sub('// _ //', '', sentence)
    sentence = re.sub('ㅋ.ㅋ', '', sentence)
    sentence = re.sub('\(--\)\(__\)', '', sentence)
    sentence = re.sub('💝', '❤', sentence)
    sentence = re.sub('ㅠㅅㅜ', '', sentence)
    sentence = re.sub('\:D', '', sentence)
    sentence = re.sub('\+_\+/', '', sentence)
    sentence = re.sub('\^-\^*', '', sentence)
    sentence = re.sub('ㅎ_ㅎ', '', sentence)
    sentence= re.sub('-_-', '', sentence)
    # sentence = re.sub('', '', sentence)
    
    # 해시태그 바꾸기
    sentence = re.sub('#', ',', sentence)
    
    # 👍🏻 👌 🤡👠 🎵 🍰🎂 🙋🏻 🙏🏻 𖤐➰ 🌹💋😲🖒💆‍♀😡👌 😴💧🙆‍♂ 😺🙆‍♂💆🏻‍♀🙆🏻🌻😮🐥🌝 \\ dev데이터셋
    # 🌹 👦🏼 👍🏻👏🏻🤘💡🍼 😲🙃🐱 🕺💝🕷🕸🏃‍♀✌🏻 💋💄📸💯💋👌🚗💬 🤮🎵🍎➰ 👆💎 🍷😜 🙆‍♂🖐💧🙋🏻‍♀ // train
    # 이모지 통일/감소
    sentence = re.sub('👍🏻', ' ', sentence)
    sentence = re.sub('😄', ' ', sentence)
    sentence = re.sub('🖒', ' ', sentence)
    sentence = re.sub('👌', '  ', sentence)
    sentence = re.sub('🤡', '  ', sentence)
    sentence = re.sub('👠', '  ', sentence)
    sentence = re.sub('🎵', '  ', sentence)
    sentence = re.sub('🍰', '  ', sentence)
    sentence = re.sub('🎂', '  ', sentence)
    sentence = re.sub('🙋🏻', '  ', sentence)
    sentence = re.sub('🙏🏻', '  ', sentence)
    sentence = re.sub('𖤐', '  ', sentence)
    sentence = re.sub('➰', '  ', sentence)
    sentence = re.sub('🌹', '  ', sentence)
    sentence = re.sub('💋', '  ', sentence)
    sentence = re.sub('😲', '  ', sentence)
    sentence = re.sub('💆‍♀', '  ', sentence)
    sentence = re.sub('😡', '  ', sentence)
    sentence = re.sub('😴', '  ', sentence)
    sentence = re.sub('💧', '  ', sentence)
    sentence = re.sub('🙆‍♂', '  ', sentence)
    sentence = re.sub('😺', '  ', sentence)
    sentence = re.sub('💆🏻‍♀', '  ', sentence)
    sentence = re.sub('🙆🏻', '  ', sentence)
    sentence = re.sub('🌻', '  ', sentence)
    sentence = re.sub('😮', '  ', sentence)
    sentence = re.sub('🐥', '  ', sentence)
    sentence = re.sub('🌝', '  ', sentence)
    sentence = re.sub('👦🏼', '  ', sentence)
    sentence = re.sub('👏🏻', '  ', sentence)
    sentence = re.sub('🤘', '  ', sentence)
    sentence = re.sub('💡', '  ', sentence)
    sentence = re.sub('🍼', '  ', sentence)
    sentence = re.sub('😲', '  ', sentence)
    sentence = re.sub('🙃', '  ', sentence)
    sentence = re.sub('🐱', '  ', sentence)
    sentence = re.sub('🕺', '  ', sentence)
    sentence = re.sub('🕷', ' ', sentence)
    sentence = re.sub('🕸', ' ', sentence)
    sentence = re.sub('🏃‍♀', '  ', sentence)
    sentence = re.sub('✌🏻', '  ', sentence)
    sentence = re.sub('💯', '  ', sentence)
    sentence = re.sub('🤮', ' ', sentence)
    sentence = re.sub('😜', ' ', sentence)
    sentence = re.sub('🖐', ' ', sentence)
    
    sentence=repeat_normalize(sentence, num_repeats=2)      
    print(sentence)
    
    input()
                
    return sentence

def replace_htag(sentence):
    # 해시태그 바꾸기
    sentence = re.sub('#', ', ', sentence)
    return sentence
