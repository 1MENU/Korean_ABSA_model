import re

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
    
    # punct 2
    
    sentence = re.sub('\,\,\,\,','...', sentence)
    sentence = re.sub('\,\,\,','...', sentence)
    sentence = re.sub('\,\,','..',sentence)
    
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

repeatchars_pattern = re.compile('(\w)\\1{3,}')
doublespace_pattern = re.compile('\s+')

def repeat_del(sentence, n): #의미없는 반복 제거 함수
    if n > 0:
        sentence = repeatchars_pattern.sub('\\1' * n, sentence)
    sentence = doublespace_pattern.sub(' ', sentence)

    return sentence.strip()

def replace_htag(sentence, to): # annotation 해시 제거 용 
    # 해시태그 바꾸기
    sentence = re.sub('#', to, sentence)
    return sentence


# 이모티콘 제거 전에 실행할 것
def replace_stars(sentence):
    # ★☆☆☆☆
    sentence = re.sub('★☆☆☆☆',' 대실망', sentence)
    # ★★☆☆☆
    sentence = re.sub('★★☆☆☆',' 실망', sentence)
    # ★★★☆☆
    sentence = re.sub('★★★☆☆',' 그럭저럭', sentence)
    # ★★★★☆
    sentence = re.sub('★★★★☆',' 만족', sentence)
    # ★★★★★
    sentence = re.sub('★★★★★',' 대만족', sentence)
    
    return sentence


def replace_unknown_token(sentence):
    
    sentence = re.sub('ㅠ', '', sentence)
    sentence = re.sub('ㅜ', '', sentence)
    # ... -> .
    sentence = re.sub('…',"...", sentence)
    sentence = re.sub('\( ◍˃̵㉦˂̵◍ \)', '', sentence)
    sentence = re.sub('ꈍ◡ꈍ', '', sentence)
    sentence = re.sub('“', '``', sentence)
    sentence = re.sub('”', '``', sentence)
    sentence = re.sub('⏰', '', sentence)
    sentence = re.sub('Ⅱ', '', sentence)
    sentence = re.sub('ɢᴇᴛ', 'get', sentence)
    sentence = re.sub('ˇ͈ᵕˇ͈', '', sentence)
    sentence = re.sub('‼', '!!', sentence)
    sentence = re.sub('ᴍᴜsᴛ', 'must', sentence)
    sentence = re.sub('ʜᴀᴠᴇ', 'have', sentence)
    sentence = re.sub('ɪᴛᴇᴍ', 'item', sentence)
    sentence = re.sub('ɴᴏɴᴏ', 'nono', sentence)
    sentence = re.sub('‼', '!!', sentence)
    sentence = re.sub('ˇ͈ᵕˇ͈', '', sentence)
    sentence = re.sub('⁉', '!?', sentence)
    sentence = re.sub('◡̈\*', '', sentence)
    sentence = re.sub('•', '', sentence)
    sentence = re.sub('°', '도', sentence)
    sentence = re.sub('‘', '`', sentence)
    sentence = re.sub('’', '`', sentence)
    sentence = re.sub('◡̈', '', sentence)
    sentence = re.sub('꺄아아아ㅏㅏㅏㅏ', '', sentence)
    sentence = re.sub('ᴜᴘ', 'up', sentence)
    sentence = re.sub('乃', '좋아요', sentence)
    sentence = re.sub('®', '', sentence)
    sentence = re.sub('◈', '', sentence)
    sentence = re.sub('→', '다음', sentence)
    sentence = re.sub('˚', '도', sentence)
    sentence = re.sub('℃', '도', sentence)
    sentence = re.sub('▲', '', sentence)
    
    return sentence

def my_preprocessing(form):
    
    form = replace_unknown_token(form)
    
    # 별점 정보 살리기
    form = replace_stars(form)
    
    # 이모티콘 제거 
    form = del_emoji_all(form)
    
    # 반복제거
    form = repeat_del(form, n=4)
    
    # punct 반복 fix
    form = preprocess_texticon(form)
    
    return form