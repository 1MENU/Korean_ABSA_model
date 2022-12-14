entity_property_pair = [
    
    '본품#품질', '제품 전체#일반', '제품 전체#품질', '본품#일반', '제품 전체#디자인',
    '본품#편의성', '제품 전체#편의성', '제품 전체#인지도', '패키지/구성품#디자인', '브랜드#일반',
    '제품 전체#가격', '패키지/구성품#편의성', '패키지/구성품#일반', '본품#다양성', '본품#디자인',
    '브랜드#품질', '패키지/구성품#품질', '브랜드#인지도', '브랜드#가격', '패키지/구성품#다양성',
    '제품 전체#다양성', '본품#가격', '브랜드#디자인', '패키지/구성품#가격', '본품#인지도'
    
]   # data label 분포도 순서


label_id_to_name = ['False', 'True']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}

polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}


special_tokens_dict = {
    'additional_special_tokens': [
        '&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&'
    ]
}