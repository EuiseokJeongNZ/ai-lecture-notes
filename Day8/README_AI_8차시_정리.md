# 강의_9기_AI개론_8차시_정리  
## 이진분류 · BCE · 평가 지표 · 실무형 분류 파이프라인

## 1. 개요

이번 강의는 **이진 분류 Binary Classification**를 다룬다.

이진 분류는 데이터를 두 그룹 중 하나로 나누는 문제이다.

예시는 다음과 같다.

```text
정상 거래 / 사기 거래
저위험 / 고위험
고객 유지 / 고객 이탈
정상 메일 / 스팸 메일
```

회귀가 연속적인 숫자를 예측하는 문제라면, 이진 분류는 0 또는 1 중 하나를 예측하는 문제이다.

핵심 구조는 다음과 같다.

```text
입력 데이터 → 선형 함수 → Sigmoid 또는 BCEWithLogitsLoss → 0/1 예측
```

---

## 2. 오늘 배운 핵심 개념

| 개념 | 설명 |
|---|---|
| 이진 분류 | 데이터를 0 또는 1 두 그룹 중 하나로 나누는 문제 |
| Sigmoid | 출력을 0~1 사이 확률로 바꾸는 함수 |
| BCELoss | Sigmoid를 지난 확률값에 사용하는 이진 분류 손실 |
| BCEWithLogitsLoss | raw logit에 직접 사용하는 권장 손실 함수 |
| Accuracy | 전체 중 맞춘 비율 |
| Precision | 양성이라고 예측한 것 중 실제 양성 비율 |
| Recall | 실제 양성 중 모델이 잡아낸 비율 |
| F1-score | Precision과 Recall의 균형 |
| Confusion Matrix | TP, TN, FP, FN을 보여주는 표 |
| ROC Curve | 임계값 변화에 따른 TPR/FPR 곡선 |
| ROC-AUC | ROC 곡선 아래 면적 |
| pos_weight | 양성 클래스에 더 큰 손실 가중치를 주는 옵션 |
| Threshold | 확률을 class로 바꾸는 기준값 |
| Pipeline | 전처리와 모델을 하나로 묶는 구조 |
| ColumnTransformer | 열 타입별로 다른 전처리를 적용하는 도구 |
| OneHotEncoder | 범주형 변수를 숫자 벡터로 바꾸는 도구 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_8차시_01_bi_cls_ok.ipynb` | Iris 이진 분류, BCELoss, BCEWithLogitsLoss, 결정 경계 |
| `강의_9기_AI개론_8차시_02_사기거래탐지_ok.ipynb` | 클래스 불균형, pos_weight, Confusion Matrix, ROC-AUC |
| `강의_9기_AI개론_8차시_03_당뇨병예측_실무형.ipynb` | Config, DataPreprocessor, DataLoader, Trainer, EarlyStopping |
| `강의_9기_AI개론_8차시_04_고객 이탈 예측(Customer Churn Prediction).ipynb` | Pipeline, ColumnTransformer, threshold tuning, feature importance |
| `08. 이진분류 (BCE 와 평가 지표).srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_8차시_(이진분류).pdf` | 이진분류 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. Sigmoid 함수 확인
3. Iris 데이터 불러오기
4. 이진 분류용 데이터 추출
5. train_test_split으로 데이터 분할
6. NumPy 배열을 Tensor로 변환
7. BCELoss 방식 모델 정의
8. BCELoss로 학습
9. BCEWithLogitsLoss 방식 모델 정의
10. BCEWithLogitsLoss로 학습
11. 결정 경계 시각화
12. 사기 거래 탐지 데이터 생성
13. stratify로 클래스 비율 유지하며 분할
14. StandardScaler로 표준화
15. pos_weight로 소수 클래스 보정
16. Confusion Matrix, Precision, Recall, F1, ROC-AUC 계산
17. Diabetes 실무형 분류 코드 구성
18. Config, DataPreprocessor, Model, Trainer, EarlyStopping 분리
19. 고객 이탈 예측용 Pipeline 구성
20. 임계값 최적화
21. Feature Importance 확인
```

---

## 5. 주요 코드 블록 설명

### 5.1 Sigmoid

```python
torch.sigmoid(x)
```

입력값을 0과 1 사이 값으로 바꾼다.

이진 분류에서는 class 1일 확률처럼 해석할 수 있다.

---

### 5.2 train_test_split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.3,
    random_state=42,
    stratify=y_data
)
```

데이터를 학습용과 테스트용으로 나눈다.

`stratify=y_data`는 클래스 비율을 유지한다.

---

### 5.3 Tensor 변환

```python
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float().view(-1, 1)
```

PyTorch 학습을 위해 NumPy 배열을 Tensor로 바꾼다.

BCE 계열 손실을 위해 정답 shape을 `[N, 1]`로 맞춘다.

---

### 5.4 BCELoss 방식

```python
self.sigmoid = nn.Sigmoid()
criterion = nn.BCELoss()
```

모델 안에서 Sigmoid를 적용하고, 확률값을 BCELoss에 넣는다.

---

### 5.5 BCEWithLogitsLoss 방식

```python
criterion = nn.BCEWithLogitsLoss()
```

모델은 raw logit만 출력하고, 손실 함수 내부에서 Sigmoid와 BCE를 안정적으로 처리한다.

실무에서는 이 방식이 권장된다.

---

### 5.6 예측 기준

```python
pred = (logits >= 0.0).float()
prob = torch.sigmoid(logits)
```

BCEWithLogitsLoss를 사용할 때는 logit 기준 0 이상이면 class 1이다.

확률이 필요할 때만 Sigmoid를 적용한다.

---

### 5.7 pos_weight

```python
pos_weight = torch.tensor([normal_count / fraud_count])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

클래스 불균형에서 양성 클래스 손실을 더 크게 반영한다.

---

### 5.8 Confusion Matrix

```python
cm = confusion_matrix(y_true, y_pred)
```

예측 결과를 TP, TN, FP, FN으로 나누어 확인한다.

---

### 5.9 Precision / Recall / F1

```python
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
```

불균형 데이터에서는 accuracy보다 더 중요한 평가 지표가 될 수 있다.

---

### 5.10 ROC-AUC

```python
auc = roc_auc_score(y_true, y_prob)
```

임계값 전체에서 모델이 class 0과 class 1을 얼마나 잘 구분하는지 확인한다.

---

### 5.11 DataLoader

```python
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Tensor 데이터를 mini-batch 단위로 꺼내 학습한다.

---

### 5.12 Pipeline

```python
pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", clf)
])
```

전처리와 모델을 하나로 묶는다.

---

### 5.13 ColumnTransformer

```python
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])
```

수치형 열과 범주형 열에 서로 다른 전처리를 적용한다.

---

### 5.14 Threshold Tuning

```python
y_hat = (y_proba >= threshold).astype(int)
```

확률을 0/1 class로 바꾸는 기준값을 조정한다.

서비스 목적에 따라 Precision과 Recall의 균형을 바꿀 수 있다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 의미 | 사용법 |
|---|---|---|
| `Sigmoid` | 0~1 확률 변환 | `torch.sigmoid(x)` |
| `logit` | Sigmoid 전 원시 점수 | `model(x)` 출력 |
| `BCE` | Binary Cross Entropy | 이진 분류 손실 |
| `BCELoss` | 확률값용 BCE | `nn.BCELoss()(prob, y)` |
| `BCEWithLogitsLoss` | logit용 BCE | `nn.BCEWithLogitsLoss()(logit, y)` |
| `Accuracy` | 전체 중 맞춘 비율 | `accuracy_score(y, pred)` |
| `Precision` | 양성 예측 중 실제 양성 비율 | `precision_score(y, pred)` |
| `Recall` | 실제 양성 중 잡은 비율 | `recall_score(y, pred)` |
| `F1` | Precision/Recall 균형 | `f1_score(y, pred)` |
| `ROC-AUC` | 임계값 전체 성능 | `roc_auc_score(y, prob)` |
| `threshold` | 분류 기준값 | `prob >= threshold` |
| `pos_weight` | 양성 클래스 가중치 | `BCEWithLogitsLoss(pos_weight=...)` |
| `stratify` | 클래스 비율 유지 | `train_test_split(..., stratify=y)` |
| `StandardScaler` | 표준화 | `fit_transform`, `transform` |
| `DataLoader` | 미니배치 생성 | `DataLoader(dataset, batch_size=...)` |
| `Pipeline` | 전처리+모델 묶음 | `Pipeline([...])` |
| `ColumnTransformer` | 열별 전처리 | 수치형/범주형 분리 |
| `OneHotEncoder` | 범주형 인코딩 | `OneHotEncoder(handle_unknown="ignore")` |

---

## 7. 그래프/출력 결과 해석

### 7.1 Sigmoid 그래프

입력이 0일 때 출력은 0.5이다.

0.5를 기준으로 class 0/1을 나눌 수 있다.

### 7.2 Iris 산점도

두 class가 feature 공간에서 어떻게 분리되는지 보여준다.

### 7.3 학습 곡선

Loss가 감소하고 Accuracy가 증가하면 학습이 진행되고 있다는 뜻이다.

### 7.4 결정 경계

로지스틱 회귀가 학습한 class 0과 class 1을 나누는 직선이다.

### 7.5 Confusion Matrix

어떤 class를 잘 맞추고 어떤 class를 놓치는지 보여준다.

### 7.6 ROC Curve

임계값 변화에 따른 모델의 분류 성능을 보여준다.

### 7.7 Precision-Recall Curve

불균형 데이터에서 양성 class를 얼마나 잘 잡는지 확인하는 데 유용하다.

---

## 8. 실습에서 배운 점

- 이진 분류는 두 class 중 하나를 예측하는 문제이다.
- Sigmoid는 선형 출력을 확률로 바꾼다.
- BCELoss를 쓸 때는 모델 마지막에 Sigmoid가 필요하다.
- BCEWithLogitsLoss를 쓸 때는 모델 마지막에 Sigmoid를 붙이면 안 된다.
- BCEWithLogitsLoss는 수치적으로 안정적이므로 실무에서 권장된다.
- 클래스 불균형 데이터에서는 Accuracy만 보면 위험하다.
- 사기 거래 탐지에서는 Recall, Precision, F1, ROC-AUC가 중요하다.
- pos_weight는 소수 class에 더 큰 손실 가중치를 준다.
- 실무형 코드는 Config, Preprocessor, Model, Trainer, Evaluator로 나누면 관리하기 쉽다.
- Pipeline과 ColumnTransformer는 수치형/범주형이 섞인 데이터에 적합하다.
- threshold는 서비스 목적에 맞게 조정해야 한다.

---

## 9. 시험용 요약

```text
이진 분류 = 0 또는 1 중 하나를 예측하는 문제
```

- Sigmoid는 출력을 0~1 확률로 바꾼다.
- Sigmoid 값이 0.5 이상이면 class 1로 예측할 수 있다.
- BCELoss는 확률값을 입력으로 받는다.
- BCEWithLogitsLoss는 raw logit을 입력으로 받는다.
- BCEWithLogitsLoss를 쓸 때 모델 마지막에 Sigmoid를 붙이지 않는다.
- BCEWithLogitsLoss 기준 예측은 logit 0 이상이면 class 1이다.
- 정확도는 전체 중 맞춘 비율이다.
- 불균형 데이터에서는 정확도만 보면 위험하다.
- Precision은 양성 예측 중 실제 양성 비율이다.
- Recall은 실제 양성 중 모델이 잡아낸 비율이다.
- F1-score는 Precision과 Recall의 균형이다.
- ROC-AUC는 임계값 전체에서의 분류 성능이다.
- Confusion Matrix는 TN, FP, FN, TP를 보여준다.
- pos_weight는 양성 class 손실을 크게 반영한다.
- stratify는 train/test의 class 비율을 유지한다.
- threshold는 목적에 따라 조정할 수 있다.