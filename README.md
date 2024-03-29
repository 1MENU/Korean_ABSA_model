# Korean_ABSA_model
2022 국립국어원 인공 지능 언어 능력 평가 (속성 기반 감성 분석 : ABSA)
<p align="center"><img src="https://github.com/1MENU/Korean_ABSA_model/blob/main/Image/leader_borad.png?raw=true" height =70% width=70%></p>
[인공지능 언어 능력 평가 대상에 '가천대학교 1인 1메뉴' 팀] 뉴스 기사

(http://www.edujin.co.kr/news/articleView.html?idxno=40816)

### 결과물 발표 영상 및 시상 (Click)

<a href="https://youtu.be/YWelwWwNhpg" target="_blank">
 <img src="http://img.youtube.com/vi/YWelwWwNhpg/maxresdefault.jpg" alt="Watch the video" width="800" height="450" border="10" />
</a>


## Task
### 속성 기반 감성 분석 (ABSA)
Task 설명 : https://corpus.korean.go.kr/task/taskList.do?taskId=8&clCd=END_TASK&subMenuId=sub01

감성 분석이란 화자의 의견, 긍/부정의 태도가 나타나는 문장의 감성 극성을 분석하여 정량화하는 것이다. 주로 문장 단위로 긍정, 부정의 유무 혹은 정도를 평가해 왔으며, 제품 및 여행의 리뷰 분석, 추천 시스템에 널리 활용되고 있는 추세이다.

최근 감성 분석 연구자들은 제품 후기 등에 전반적으로 나타나는 하나의 감성 극성(긍정 또는 부정)을 추출하는 것을 넘어 더 세부적이고 구체적인 수준의 감성 관련 정보를 추출하는 데 관심을 갖기 시작했다. 속성 기반 감성 분석은 언어 자료에 나타난 개체와 속성 정보를 고려한 감성 분석 방법으로 일반적인 감성 분석에 비해 더 세밀한 정보를 얻을 수 있다는 장점이 있다. 예를 들어, 음식점 도메인의 리뷰 “콩국수가 싸서 좋다”라는 문장에서 일반적인 감성 분석의 결과는 “긍정”이 되지만, 속성 기반 감성 분석에서는 개체:{음식(콩국수)}, 속성{가격}, 감성:{긍정}으로 더 많고 구체적인 정보를 얻을 수 있다. 
<p align="center"><img src="https://github.com/1MENU/Korean_ABSA_model/blob/main/Image/task.png?raw=true" height =70% width=70%></p>


## Solution
### Dataset Preprocessing

1. train, dev data의 속성범주 및 감성 label 분포도 (Data imbalance)
<p align="center"><img src="https://github.com/1MENU/Korean_ABSA_model/blob/main/Image/annotation_lbl.png?raw=true" height =45% width=45%>
<img src="https://github.com/1MENU/Korean_ABSA_model/blob/main/Image/sentiment_lbl.png?raw=true" height =20% width=20%>
</p>


2. 정제되지 않은 Data (비문, 줄임말, 오타, 텍스티콘 등)
> 리뷰 데이터도 포함된 data로 pretrain을 진행했기 때문에 오타 및 비문, 줄임말에 유리하지만 이모지, 라틴어, 특수문자 등은 단어장에 포함되어있지 않는 kykim/electra-kor-base의 특징에 따라 전처리 진행
3. Data label의 낮은 정확도

 > * 약 5,800건 가량의 train, dev 데이터에서 라벨 분류가 애매모호한 문장 삭제 (약 15문장)
 > * 같은 속성 범주 내 다른 감성분석 주석이 있는 경우 이를 제거 혹은 수정
 > * Label Smoothing을 통해 부정확한 데이터에 대한 보정 효과를 주고, 모델의 over-confident도 줄임


### ACD & ASC
Aspect Category Detection (ACD) 모델과 Aspect Sentiment Classification (ASC) 모델을 연결하였다. 학습 데이터가 충분하지 않기 때문에 transfer learning을 이용해 기존의 pretrained model 모델에 독자적인 classification head를 추가한 후 fine-tuning하여 학습하였다.

<p align="center"><img src="https://github.com/1MENU/Korean_ABSA_model/blob/main/Image/model_architecture.png?raw=true" height =70% width=70%></p>

> 1. 전처리와 tokenizing을 거친 token들을 “[CLS] 문장 [SEP] 속성범주 [SEP]”의 형태로 pretrained-model input으로 사용한다.
> 2. output layer들의 내부 Self-attention Layer 12개 [CLS] token들을 attention pooling 해 256차원의 벡터를 만든다.
> 3. last hidden layer의 속성범주 토큰 부분만 추출해 768차원의 벡터를 만든다.
> 4. 각각의 다른 FC layer[dropout(0.1), activation function으로 hyperbolic tangent, Linear]를 거친다.
> 5. 2개의 벡터를 concat하여 1024차원의 새로운 벡터를 만든다.
> 6. 이를 분류 벡터를 사용하여 1024->2 linear를 적용한다.

결과가 0(False)일시 해당 속성범주는 추출되지 않고, 1(True)일시 속성범주가 추출된다.

ASC 또한 같은 모델 구조를 지니고 있으며, Class label이 0(positive), 1(negative), 2(neutral)로 이루어져있어 분류벡터에서 1024 -> 3 linear를 적용하는 것만 제외하면 전부 동일하다. 

아래는 본 모델의 결과이다.

<p align="center"><img src="https://github.com/1MENU/Korean_ABSA_model/blob/main/Image/result.png?raw=true" height =70% width=70%></p>

## Implement
1. git clone (https://github.com/1MENU/Korean_ABSA_model.git)
2. Download the model (https://huggingface.co/juicyjung/Korean_ABSA_model_1MENU)
> huggingface에서 가져온 파일을 압축해제하여 materials/saved_model 폴더 아래에 넣기
> CD 폴더 안의 파일은 CD 폴더 아래에, SC 폴더 안의 파일은 SC 폴더 아래에 넣는다.

3. 모델에 넣을 dataset마련
* 실행시 dataset 폴더 아래에다가 inference 할 데이터 넣기
* 이때 설정할 파일의 이름은 run_together.sh 파일 내, —test_file값에 실행하고자하는 파일의 이름과 같아야한다. (기본값은 “nikluge-sa-2022-test.jsonl” 이다)

4. docker file를 이용하여 환경 구축하기 

```zsh
docker build -t gcu-1menu:1.0 . 
docker run –it —name team1 gcu-1menu:1.0
```
5.  run_together.sh 실행하기
```zsh
bash run_together.sh 
```
- 결과 파일은 material/submission 폴더 아래 생성된다. 현재 final.json이라는 이름으로 결과값이 나오게 되어있다
- 다만 주의해야할 점은 결과파일은 도커 내부 환경에만 반영이 되어있으므로 도커 컨테이너 환경에서 결과 파일을 가져오고 싶은 경우에는 아래의 명령어를 사용한다.
```zsh
docker cp <컨테이너 이름>:<컨테이너 내부 파일 경로> <복사할 파일 경로> 
```

## Members

Jiwoo Jung | travelandi01@gmail.com<br>
Doyeon Hyun | 118ssun@naver.com<br>
Seonghyun Kang | manomono0610@gmail.com<br>
Heejin Jang | heejin00628@gmail.com<br>
Hajeong Lee | hjpurege@gachon.ac.kr
