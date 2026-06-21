# 강의_9기_AI개론_17차시_정리  
## 자연어처리 NLP 종합실습 · BiLSTM · DistilBERT · Tokenizer · Transformer

## 1. 개요

이번 17차시는 기존 코드 재활용과 자연어처리 NLP 종합 실습을 정리하는 강의다.

앞 강의까지는 이미지 데이터를 중심으로 CNN, pretrained model, custom dataset을 다뤘다.  
이번 강의에서는 데이터 형태가 이미지에서 텍스트로 바뀐다.

강의에서 강조한 핵심 구조는 다음이다.

```text
이미지 분류: image, label
자연어 분류: text, label
```

문제 구조는 비슷하다.  
하지만 텍스트는 바로 Tensor가 아니므로 모델에 넣기 전에 반드시 숫자로 바꾸어야 한다.

핵심 흐름은 다음이다.

```text
자연어처리 NLP 큰 그림
→ 텍스트 전처리
→ 토큰화 Tokenization
→ 어휘사전 Vocab
→ 인덱스 변환 Numericalization
→ 패딩 Padding
→ 임베딩 Embedding
→ RNN/LSTM/BiLSTM
→ IMDb 감성 분류
→ DistilBERT 파인튜닝
→ Trainer 사용
→ 평가 지표 accuracy, precision, recall, f1
→ Attention, Transformer, BERT, GPT 개념 정리
→ 과제 빈칸 패턴 정리
```

이번 정리는 두 가지 실행 환경으로 나누었다.

```text
인터넷 가능 버전
→ Hugging Face IMDb dataset, BiLSTM, DistilBERT를 사용한다.

인터넷 불가 버전
→ 직접 만든 문장 데이터로 tokenizer, vocab, padding, LSTM/BiLSTM 구조를 연습한다.
```

강의 원본 흐름은 거의 online 버전 기준이다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| NLP | Natural Language Processing, 자연어처리 |
| Text Classification | 문장을 class로 분류하는 문제 |
| Tokenization | 문장을 작은 token 단위로 자르는 작업 |
| Vocab | token과 숫자 id를 연결하는 어휘사전 |
| Numericalization | token을 숫자 id로 바꾸는 과정 |
| Padding | sequence 길이를 맞추기 위해 `<pad>`를 채우는 과정 |
| Truncation | 너무 긴 문장을 최대 길이에서 자르는 과정 |
| Embedding | token id를 dense vector로 바꾸는 layer |
| RNN | sequence를 순서대로 처리하는 순환 신경망 |
| LSTM | 장기 의존성 문제를 완화한 RNN 계열 모델 |
| BiLSTM | 문장을 앞뒤 양방향으로 읽는 LSTM |
| Attention | 중요한 token에 더 집중하는 구조 |
| Transformer | Attention 중심으로 sequence를 처리하는 구조 |
| BERT | Transformer Encoder 기반의 문맥 이해 모델 |
| GPT | Transformer Decoder 기반의 생성 모델 |
| DistilBERT | BERT를 경량화한 모델 |
| DataCollatorWithPadding | batch별 동적 padding 도구 |
| Trainer | Hugging Face 학습 루프 자동화 도구 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `Summary_Day17_online_student_note_da.ipynb` | 인터넷 가능 버전, IMDb + BiLSTM + DistilBERT 중심 |
| `Summary_Day17_offline_student_note_da.ipynb` | 인터넷 불가 버전, 직접 만든 문장 데이터 + LSTM/BiLSTM 중심 |
| `Summary_Day17_student_note_da.ipynb` | 두 버전 안내용 파일 |
| `README_AI_17차시_정리_student_note_da.md` | 17차시 개념과 코드 흐름 정리 |
| `강의_9기_AI개론_17차시_01_BiLSTM 실습(초급).ipynb` | IMDb 데이터셋, tokenizer, vocab, BiLSTM 실습 |
| `강의_9기_AI개론_17차시_02_nlp_advanced_DistilBERT_finetuned.ipynb` | DistilBERT tokenizer/model, Trainer fine-tuning 실습 |
| `과제_9기_AI개론_17차시_본인이름.ipynb` | BasicLSTM, DataCollator, BiLSTM hidden 연결 빈칸 과제 |
| `[보강] w2v,tokenizer_ok.ipynb` | Korpora, NSMC, tokenizer, Word2Vec 보강 |
| `[보강] LSTM.ipynb` | LSTM 구조 보강 |
| `[보강] NLP SPECIAL.ipynb` | TF-IDF, 코사인 유사도 보강 |
| `강의_9기_AI개론_17차시_(자연어처리_NLP_종합실습).pdf` | NLP 전체 흐름 자료 |
| `RNN 기반으로 하는 자연어처리.pdf` | RNN, LSTM, GRU, Word2Vec 자료 |
| `Attention 과 Transformer.pdf` | Attention과 Transformer 자료 |
| `BERT.pdf` | BERT 자료 |
| `GPT.pdf` | GPT 자료 |
| `17. 종합 실습, 리뷰(기존 코드 재활용).srt` | 강의 스크립트 |

---

## 4. 코드 흐름 요약

### 인터넷 가능 버전

```text
1. 라이브러리 import
2. device와 seed 설정
3. NLP 전체 그림 정리
4. 정규식 기반 simple_tokenize 함수 정의
5. Hugging Face IMDb dataset 로드
6. Counter로 token 빈도수 계산
7. PAD, UNK 특수 token 정의
8. itos, stoi 어휘사전 생성
9. encode 함수로 문장을 token id sequence로 변환
10. padding/truncation 적용
11. IMDBTensor Dataset 정의
12. DataLoader 구성
13. Embedding layer shape 확인
14. BiLSTM 모델 정의
15. forward 방향과 backward 방향 hidden state 연결
16. BiLSTM 학습/평가 함수 정의
17. BiLSTM 짧은 학습 실행
18. BasicLSTM 과제 빈칸 패턴 정리
19. BiLSTM hidden 연결 과제 패턴 정리
20. DistilBERT 개념 정리
21. AutoTokenizer와 AutoModelForSequenceClassification 로드
22. preprocess 함수로 tokenizer 적용
23. DataCollatorWithPadding으로 동적 padding
24. compute_metrics 함수 정의
25. TrainingArguments 버전 호환 생성 함수 정의
26. Trainer 구성과 학습 흐름 정리
27. DistilBERT 추론 예시
28. Attention, Transformer, BERT, GPT 큰 그림 정리
```

### 인터넷 불가 버전

```text
1. 라이브러리 import
2. 긍정/부정 영화 리뷰 문장 직접 생성
3. 정규식 tokenizer 정의
4. Counter로 vocab 생성
5. stoi, itos 생성
6. encode 함수로 token id 변환과 padding 적용
7. TinyReviewDataset 정의
8. DataLoader 구성
9. BasicLSTM 모델 정의
10. TinyBiLSTM 모델 정의
11. 학습 함수 정의
12. 작은 데이터 학습 실행
13. 새 문장 추론
14. DistilBERT 개념만 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 토큰화

```python
token_pattern = re.compile(r"[a-z0-9']+")

def simple_tokenize(text):
    return token_pattern.findall(text.lower())
```

문장을 소문자로 바꾼 뒤 영어 단어/숫자 token을 추출한다.

---

### 5.2 Vocab 만들기

```python
counter = Counter()

for ex in raw["train"]:
    counter.update(simple_tokenize(ex["text"]))

itos = [PAD, UNK] + [t for t, _ in counter.most_common(MAX_VOCAB - 2)]
stoi = {t: i for i, t in enumerate(itos)}
```

token 빈도수를 세고, token과 id를 연결하는 사전을 만든다.

---

### 5.3 encode 함수

```python
ids = [stoi.get(tok, UNK_IDX) for tok in tokens][:MAX_LEN]

if len(ids) < MAX_LEN:
    ids += [PAD_IDX] * (MAX_LEN - len(ids))
```

문장을 token id sequence로 바꾸고, 최대 길이에 맞춰 자르거나 padding한다.

---

### 5.4 PyTorch Dataset

```python
class IMDBTensor(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        x = torch.tensor(encode(text), dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
```

원본 text와 label을 DataLoader가 사용할 수 있는 Tensor 형태로 바꾼다.

---

### 5.5 Embedding

```python
self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_idx)
```

token id를 dense vector로 바꾼다.

입력 shape은 `[batch, seq_len]`이고 출력 shape은 `[batch, seq_len, emb]`이다.

---

### 5.6 BiLSTM

```python
self.lstm = nn.LSTM(
    input_size=emb,
    hidden_size=hidden,
    batch_first=True,
    bidirectional=True
)
```

문장을 앞 방향과 뒤 방향으로 읽는다.

양방향이므로 hidden state를 연결하면 `hidden * 2` 차원이 된다.

---

### 5.7 BiLSTM hidden 연결

```python
last_f = h[-2]
last_b = h[-1]
h_cat = torch.cat([last_f, last_b], dim=1)
```

마지막 layer의 forward hidden과 backward hidden을 feature 차원으로 연결한다.

---

### 5.8 BasicLSTM 과제 패턴

```python
self.embedding = nn.Embedding(vocab_size, embed_size)
self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
self.fc = nn.Linear(hidden_size, num_classes)
```

단방향 LSTM에서는 `hn[0]`을 `fc`에 넣는다.

---

### 5.9 AutoTokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
```

DistilBERT에 맞는 tokenizer를 자동으로 불러온다.

직접 vocab을 만들 필요가 없다.

---

### 5.10 AutoModelForSequenceClassification

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

문장 분류용 head가 붙은 DistilBERT 모델을 불러온다.

`num_labels=2`는 부정/긍정 2 class 분류라는 뜻이다.

---

### 5.11 DataCollatorWithPadding

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

batch 안에서 가장 긴 문장 길이에 맞춰 padding한다.

고정 길이 padding보다 메모리를 아낄 수 있다.

---

### 5.12 compute_metrics

```python
preds = logits.argmax(axis=1)
```

logits에서 가장 큰 class를 예측으로 선택하고 accuracy, precision, recall, f1을 계산한다.

---

### 5.13 Trainer

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
```

Hugging Face 방식으로 학습 루프를 자동 처리한다.

---

### 5.14 DistilBERT 추론

```python
inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
probs = torch.softmax(model(**inputs).logits, dim=-1)
```

문장을 token id로 바꾸고 모델에 넣어 부정/긍정 확률을 계산한다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `NLP` | 자연어처리 | 텍스트 이해/생성 |
| `text` | 원문 문장 | 모델 입력 원본 |
| `label` | 정답 class | 0/1 등 |
| `token` | 문장을 나눈 단위 | 단어/subword |
| `tokenizer` | token/id 변환 도구 | `AutoTokenizer` |
| `vocab` | 어휘사전 | token-id 매핑 |
| `stoi` | string to index | token → id |
| `itos` | index to string | id → token |
| `PAD` | padding token | 길이 맞춤 |
| `UNK` | unknown token | 모르는 token |
| `MAX_LEN` | 최대 sequence 길이 | padding/truncation 기준 |
| `Embedding` | token id를 벡터로 변환 | `nn.Embedding` |
| `RNN` | 순환 신경망 | sequence 처리 |
| `LSTM` | 장단기 기억 모델 | 장기 의존성 완화 |
| `BiLSTM` | 양방향 LSTM | 앞뒤 문맥 학습 |
| `hn` | hidden state | LSTM 마지막 은닉 상태 |
| `cn` | cell state | LSTM 기억 상태 |
| `logits` | class 점수 | softmax 전 값 |
| `AdamW` | Adam + weight decay | Transformer 계열에서 자주 사용 |
| `DistilBERT` | 경량 BERT | 빠른 Transformer 분류 |
| `DataCollatorWithPadding` | 동적 padding | batch별 길이 맞춤 |
| `Trainer` | HF 학습 도구 | 학습 루프 자동화 |
| `precision` | 정밀도 | 예측 양성 중 실제 양성 |
| `recall` | 재현율 | 실제 양성 중 맞힌 비율 |
| `f1` | precision/recall 조화 평균 | 분류 평가 지표 |
| `Attention` | 중요한 token에 집중 | Transformer 핵심 |
| `BERT` | Encoder 기반 모델 | 문맥 이해 |
| `GPT` | Decoder 기반 모델 | 텍스트 생성 |

---

## 7. 그래프/출력 결과 해석

### 7.1 문장 길이 분포

문장 길이 분포를 보면 `MAX_LEN`을 정하는 데 도움을 준다.

너무 짧게 잡으면 정보가 잘리고, 너무 길게 잡으면 padding이 많아진다.

### 7.2 BiLSTM Loss Curve

loss가 내려가면 모델이 정답 감성에 가까워지는 것이다.

### 7.3 BiLSTM Accuracy Curve

accuracy가 올라가면 긍정/부정을 더 잘 맞힌다는 뜻이다.

### 7.4 DistilBERT 평가 결과

DistilBERT는 accuracy뿐 아니라 precision, recall, f1을 함께 본다.

감성 분류처럼 class가 2개인 문제에서도 f1은 중요한 지표다.

### 7.5 추론 확률

Negative와 Positive 확률을 보면 모델이 어느 쪽으로 더 강하게 판단했는지 확인할 수 있다.

---

## 8. 실습에서 배운 점

- 자연어처리는 전처리가 매우 중요하다.
- 텍스트는 그대로 모델에 넣을 수 없고 token id로 바꿔야 한다.
- tokenizer는 문장을 token 단위로 나누고 숫자로 바꾸는 역할을 한다.
- padding과 truncation은 batch 학습을 위해 필요하다.
- Embedding은 token id를 의미 공간의 vector로 바꾼다.
- LSTM은 sequence 순서를 학습한다.
- BiLSTM은 앞뒤 문맥을 모두 사용한다.
- DistilBERT는 BERT보다 가볍고 빠른 Transformer 계열 모델이다.
- Attention은 중요한 token에 집중하는 구조다.
- BERT는 문맥 이해에 강하고, GPT는 생성에 강하다.
- Hugging Face Trainer를 쓰면 fine-tuning 코드가 짧아진다.
- 그래도 tokenizer, collator, metrics, training args의 역할은 알아야 한다.

---

## 9. 시험용 요약

```text
17강 핵심 = 텍스트를 숫자 sequence로 바꾸고 BiLSTM 또는 DistilBERT로 감성 분류를 수행하는 것
```

- 자연어처리는 텍스트를 컴퓨터가 계산 가능한 숫자로 바꾸는 과정이 중요하다.
- 이미지 분류는 image와 label이고, 자연어 분류는 text와 label이다.
- 토큰화는 문장을 작은 token 단위로 자르는 것이다.
- 어휘사전은 token과 숫자 id를 연결한다.
- `stoi`는 string to index다.
- `itos`는 index to string이다.
- `<pad>`는 길이를 맞추기 위한 token이다.
- `<unk>`는 사전에 없는 token을 처리하기 위한 token이다.
- padding은 짧은 문장을 같은 길이로 채우는 것이다.
- truncation은 긴 문장을 최대 길이에서 자르는 것이다.
- embedding은 token id를 dense vector로 바꾼다.
- LSTM은 sequence 순서를 처리한다.
- BiLSTM은 문장을 앞뒤 양방향으로 읽는다.
- BiLSTM의 hidden은 forward와 backward를 `torch.cat(..., dim=1)`으로 연결한다.
- DistilBERT는 BERT를 경량화한 모델이다.
- BERT 계열은 Attention 기반으로 문맥 이해에 강하다.
- GPT 계열은 다음 token 생성에 강하다.
- `DataCollatorWithPadding`은 batch마다 동적 padding을 적용한다.
- `Trainer`는 Hugging Face 모델 학습 루프를 대신 처리한다.
- accuracy만 보지 말고 precision, recall, f1도 함께 본다.