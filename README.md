# Korean_ABSA_model
2022 국립국어원 인공 지능 언어 능력 평가 (속성 기반 감성 분석 : ABSA)

## Solution
### Dataset Preprocessing

### ACD & ASC
Aspect Category Detection (ACD) 모델과 Aspect Sentiment Classification (ASC) 모델을 연결하였다. 학습 데이터가 충분하지 않기 때문에 transfer learning을 이용해 기존의 pretrained model 모델에 독자전인 classification head를 추가한 후 fine-tuning하여 학습하였다.



## Members

Jiwoo Jung | <br>
Doyeon Hyun | 118ssun@naver.com<br>
Seonghyun Kang | manomono0610@gmail.com<br>
Heejin Jang | heejin00628@gmail.com<br>
Hajeong Lee | hjpurege@gachon.ac.kr

## Reference
(Jason Wei, Kai Zou, EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks, Protago Labs Research, Tysons Corner, Virginia, USA, Department of Computer Science, Dartmouth College, Department of Mathematics and Statistics, Georgetown University ,2019, p.5) 

