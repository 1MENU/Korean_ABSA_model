import re
from soynlp.normalizer import *
from hanspell import spell_checker

def spacing_sent(sentence):
    
    def special_tok_change(sentence):
        #'&name&', '&affiliation&', '&social-security-num&', 
        # '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&'
        sentence=re.sub('&name&','$name$',sentence)
        sentence=re.sub('&affiliation&','$affiliation$',sentence)
        sentence=re.sub('&social-security-num&','$social-security-num$',sentence)
        sentence=re.sub('&tel-num&','$tel-num$',sentence)
        sentence=re.sub('&card-num&','$card-num$',sentence)
        sentence=re.sub('&bank-account&','$bank-account$',sentence)
        sentence=re.sub('&num&','$num$',sentence)
        sentence=re.sub('&online-account&','$online-account$',sentence)
        return sentence
    
    sentence=special_tok_change(sentence) # xml 파싱 시에 &에서 오류발생해서 다 바꿔주기
    sentence=re.sub('&',', ',sentence)
    
   # print("before : ", sentence)
    
    result_train = spell_checker.check(sentence)
    sentence = result_train.as_dict()['checked']
    
   # print("after : ", sentence)
    
    return sentence 


def preprocess_texticon(sentence):
    
    sentence = re.sub('~~~~~~~~', '~~~', sentence)
    sentence = re.sub('~~~~~~~', '~~~', sentence)
    sentence = re.sub('~~~~~~', '~~~', sentence)
    sentence = re.sub('~~~~~', '~~~', sentence)
    sentence = re.sub('~~~~', '~~~', sentence)
    
    sentence = re.sub('\?\?\?\?', '?', sentence)
    sentence = re.sub('\?\?\?', '?', sentence)
    sentence = re.sub('\?\?', '?', sentence)
    
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!', '!', sentence)
    sentence = re.sub('\!\!', '!', sentence)
    
    sentence = re.sub('\?\!', '?', sentence)
    sentence = re.sub('\!\?', '?', sentence)
    
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.\.', '...', sentence)
    sentence = re.sub('\.\.\.\.', '...', sentence)
    
    return sentence
    

def remove_texticon(sentence):
    # 텍스트 이모지

    sentence = re.sub('\;','', sentence)
    sentence = re.sub('\^\^', '', sentence)
    sentence = re.sub(':\)', '', sentence)
    sentence = re.sub('>.<', '', sentence)
    sentence = re.sub('ㅎㅅㅎ', '', sentence)
    sentence = re.sub(' \*\_\*','', sentence)
    sentence = re.sub(' \*ㅅ\*','',sentence)
    sentence = re.sub('> 3 <', '', sentence)
    sentence = re.sub('// _ //', '', sentence)
    sentence = re.sub('ㅋ.ㅋ', '', sentence)
    sentence = re.sub('\(--\)\(__\)', '', sentence)
    sentence = re.sub('ㅠㅅㅜ', '', sentence)
    sentence = re.sub(':D', '', sentence)
    sentence = re.sub('\+_\+/', '', sentence)
    sentence = re.sub('\^-\^*', '', sentence)
    sentence = re.sub('ㅎ_ㅎ', '', sentence)
    sentence = re.sub('-_-;', '', sentence)
    sentence = re.sub('\+\+', '', sentence)
    sentence = re.sub('- -', '', sentence)
    sentence = re.sub('`-`', '', sentence)
    sentence = re.sub('ෆ', '', sentence)
    
    sentence = re.sub('ㅋ', '', sentence)
    sentence = re.sub(' ㅋ', '', sentence)
    sentence = re.sub('ㅋㅋ', '', sentence)
    sentence = re.sub(' ㅋㅋ', '', sentence)
    sentence = re.sub(' ᄏᄏ','',sentence)
    sentence = re.sub('ㅋㅋㅋ', '', sentence)
    sentence = re.sub(' ㅋㅋㅋ', '', sentence)
    
    sentence = re.sub('ㅎ', '', sentence)    
    sentence = re.sub('ㅎㅎ','', sentence)
    sentence = re.sub('ㅎㅎㅎ', '', sentence) # ᄒᄒᄒ
    sentence = re.sub(' ᄒᄒᄒ','', sentence)
    sentence = re.sub(' ㅎ', '', sentence)    
    sentence = re.sub(' ㅎㅎ','', sentence)
    sentence = re.sub(' ㅎㅎㅎ', '', sentence)
    
    sentence = re.sub('ㅠ','', sentence)
    sentence = re.sub('ㅠㅠ','', sentence)
    sentence = re.sub('ㅠㅠㅠ','', sentence)
    sentence = re.sub(' ㅠ','', sentence)
    sentence = re.sub(' ㅠㅠ','', sentence)
    sentence = re.sub(' ㅠㅠㅠ','', sentence)
    sentence = re.sub('ㅜ_ㅜ','',sentence)
    sentence = re.sub('ㅜ','', sentence)
    sentence = re.sub('ㅜㅜ','', sentence)
    sentence = re.sub('ㅜㅜㅜ','', sentence)
    sentence = re.sub(' ㅜ','', sentence)
    sentence = re.sub(' ㅜㅜ','', sentence)
    sentence = re.sub( 'ㅜㅜㅜ','', sentence)
    
    sentence = re.sub('\( ◍˃̵㉦˂̵◍ \)', '', sentence)
    sentence = re.sub('ღ`ᴗ`ღ', '', sentence)
    sentence = re.sub('\+_\+', '', sentence)
    sentence = re.sub('‘-‘', '', sentence)
    sentence = re.sub('ㅠ_ㅠ', '', sentence)
    sentence = re.sub('>_<', '', sentence)
    sentence = re.sub('\^-\^/', '', sentence)
    sentence = re.sub('\^_\^', '', sentence)
    sentence = re.sub(':-\)', '', sentence)
    
    
    sentence = re.sub('~~~~~~~~', '~~~', sentence)
    sentence = re.sub('~~~~~~~', '~~~', sentence)
    sentence = re.sub('~~~~~~', '~~~', sentence)
    sentence = re.sub('~~~~~', '~~~', sentence)
    sentence = re.sub('~~~~', '~~~', sentence)
    
    sentence = re.sub('\?\?\?\?', '?', sentence)
    sentence = re.sub('\?\?\?', '?', sentence)
    sentence = re.sub('\?\?', '?', sentence)
    
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!\!', '!', sentence)
    sentence = re.sub('\!\!\!', '!', sentence)
    sentence = re.sub('\!\!', '!', sentence)
    
    sentence = re.sub('\?\!', '?', sentence)
    sentence = re.sub('\!\?', '?', sentence)
    
    sentence = re.sub('\,\,\,\,',' ', sentence)
    sentence = re.sub('\,\,\,',' ', sentence)
    sentence = re.sub('\,\,',' ',sentence)

    sentence = re.sub('\. \.', ' ', sentence)
    
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.\.', ' ', sentence)
    sentence = re.sub('\.\.', ' ', sentence)

    
    sentence = re.sub('ꈍ◡ꈍ','', sentence)
    sentence = re.sub('>__<','', sentence)
    sentence = re.sub('>_','', sentence)
    sentence = re.sub('……','', sentence)
    sentence = re.sub('◡̈\*','', sentence) 
    sentence = re.sub('\;\)','', sentence)
    sentence = re.sub('\+ㅁ\+','', sentence)
    sentence = re.sub(' \:\)','', sentence)
    sentence = re.sub('\*ㅁ\*','',sentence)
    sentence = re.sub('` 3`\*','',sentence)
    sentence = re.sub('><','', sentence)
    
    sentence = re.sub('▲', '', sentence)
    sentence = re.sub('@.@', '', sentence)
 
    
    #그 먼지들을 탁탁 쳐서 제대로 청소해주니까 ;) 더욱 깔끔하게 청소가 되는듯하고요 -
    #replace_marks 결과 : ​타 제품들은 입술에 바르면 바른 부분과 안 바른 부분이 그라 하기도 전에 딱 경계선이 질 정도로 발색과 착색이 강한데, 
    # 이건 물을 많이 탄 듯한 연한 워터 제형이라 야리야리한 장미 꽃잎 색깔이랄까? *_*
    
    return sentence


def del_emoji_all(sentence):
    
    pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        # u"\U0002500-\U0002BEF"  # chinese char
        # u"\U00002702-\U000027B0"
        # u"\U000024C2-\U0001F251" # 이 셋 중 하나인지 뭔지 텍스트를 싹 날려버림
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", re.UNICODE) 
        
    return re.sub(pattern, '', sentence)


def repeat_del(sentence, n): #의미없는 반복 제거 함수 
    sentence=repeat_normalize(sentence, num_repeats=n)   
    return sentence

def replace_htag(sentence, to): # annotation 해시 제거 용 
    # 해시태그 바꾸기
    sentence = re.sub('#', to, sentence)
    return sentence

def replace_marks(sentence):
    # 띄어쓰기
    #sentence ="별로네 ,, "
    sentence = spacing_sent(sentence)
    #반복제거 
    sentence = repeat_del(sentence, n=3)    
    # 텍스트 이모티콘 제거 
    sentence = remove_texticon(sentence)
    # 이모티콘 제거 
    sentence = del_emoji_all(sentence)
   
    return sentence