# 강의_9기_AI개론_8차시_정리  
## 이진 분류 · Sigmoid · BCE · BCEWithLogitsLoss · 평가 지표 · 불균형 데이터 · 실무형 분류 파이프라인

## 1. 개요

이번 8차시는 이진 분류를 정리하는 강의다.

앞에서 배운 회귀는 연속적인 숫자를 예측하는 문제였다.  
반면 이진 분류는 데이터를 두 그룹 중 하나로 나누는 문제다.

```text
회귀: 집값이 얼마인가?
이진 분류: 사기 거래인가 아닌가?
```

이번 강의의 핵심 흐름은 다음이다.

```text
이진 분류 문제 정의
→ Sigmoid로 확률 만들기
→ 0.5 threshold로 class 예측
→ BCE / BCEWithLogitsLoss로 손실 계산
→ Accuracy로 기본 성능 확인
→ Precision, Recall, F1, ROC-AUC로 더 자세히 평가
→ Confusion Matrix로 예측 결과 구조 확인
→ 클래스 불균형에서 pos_weight 사용
→ 사기 거래 탐지, 당뇨병 예측, 고객 이탈 예측으로 실무형 흐름 확장
```

이번 강의에서 가장 중요한 차이는 `BCELoss`와 `BCEWithLogitsLoss`다.  
`BCELoss`는 Sigmoid를 지난 확률값을 받고, `BCEWithLogitsLoss`는 Sigmoid 전 raw score인 logit을 직접 받는다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| 이진 분류 | 데이터를 0 또는 1 두 class 중 하나로 나누는 문제 |
| Sigmoid | logit을 0과 1 사이 확률로 바꾸는 함수 |
| Logit | Sigmoid 전 모델 raw output |
| Probability | Sigmoid를 지난 확률값 |
| Threshold | 확률을 class로 바꾸는 기준값 |
| BCE | Binary Cross Entropy, 이진 분류 손실함수 |
| BCELoss | 확률값을 입력으로 받는 BCE |
| BCEWithLogitsLoss | logit을 입력으로 받는 안정적인 BCE |
| Accuracy | 전체 중 맞춘 비율 |
| Precision | 1이라고 예측한 것 중 실제 1의 비율 |
| Recall | 실제 1 중 모델이 찾아낸 비율 |
| F1-score | Precision과 Recall의 균형 |
| Confusion Matrix | TN, FP, FN, TP를 표로 보여주는 지표 |
| ROC-AUC | threshold 전체에서 모델의 분리 능력을 보는 지표 |
| Class Imbalance | 특정 class가 매우 적은 데이터 불균형 상황 |
| pos_weight | BCEWithLogitsLoss에서 양성 class에 주는 가중치 |
| EarlyStopping | validation loss가 개선되지 않으면 학습을 멈추는 기법 |
| Pipeline | 전처리와 모델을 하나로 묶는 scikit-learn 구조 |
| Threshold Optimization | 목적 지표에 맞게 임계값을 조정하는 과정 |
| Permutation Importance | feature를 섞었을 때 성능 하락으로 중요도를 보는 방법 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_8차시_01_bi_cls_ok.ipynb` | Iris 이진 분류, Sigmoid, BCELoss, BCEWithLogitsLoss, 결정 경계 |
| `강의_9기_AI개론_8차시_02_사기거래탐지_ok.ipynb` | 클래스 불균형, 사기 거래 탐지, pos_weight, Confusion Matrix, ROC-AUC |
| `강의_9기_AI개론_8차시_03_당뇨병예측_실무형.ipynb` | Config, DataPreprocessor, BatchNorm/Dropout 모델, EarlyStopping, PR Curve |
| `강의_9기_AI개론_8차시_04_고객 이탈 예측(Customer Churn Prediction).ipynb` | 고객 이탈 합성 데이터, ColumnTransformer, Pipeline, Threshold Optimization, Feature Importance |
| `08. 이진분류 (BCE 와 평가 지표).srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_8차시_(이진분류).pdf` | 이진 분류, Sigmoid, 교차 엔트로피, 평가 지표 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. 이진 분류 개념 정리
3. Sigmoid 함수 그래프 확인
4. logit, probability, threshold 관계 정리
5. BCE 직관 확인
6. BCELoss 사용법 확인
7. BCEWithLogitsLoss 사용법 확인
8. Iris 데이터 불러오기
9. 앞의 두 class와 두 feature만 선택
10. train/validation 분할
11. 산점도로 두 class 분포 확인
12. Tensor 변환
13. Linear + Sigmoid 모델 정의
14. BCELoss 방식으로 학습
15. loss/accuracy curve 확인
16. Linear only 모델 정의
17. BCEWithLogitsLoss 방식으로 학습
18. 결정 경계 시각화
19. Accuracy, Precision, Recall, F1 계산
20. Confusion Matrix 시각화
21. ROC Curve와 AUC 계산
22. 사기 거래 탐지용 불균형 데이터 생성
23. stratify 분할과 StandardScaler 적용
24. FraudDetectionModel 정의
25. pos_weight 계산
26. BCEWithLogitsLoss(pos_weight=...)로 학습
27. Classification Report, Confusion Matrix, ROC-AUC 확인
28. 당뇨병 예측 실무형 Config 정의
29. DataPreprocessor로 데이터 준비와 DataLoader 생성
30. DiabetesClassifier 모델 정의
31. EarlyStopping 정의
32. Trainer 클래스로 학습 루프 구성
33. Loss, Accuracy, Learning Rate 그래프 확인
34. Precision-Recall Curve 확인
35. 고객 이탈 합성 데이터 생성
36. 수치형/범주형 feature 분리
37. ColumnTransformer와 Pipeline 구성
38. LogisticRegression(class_weight="balanced") 학습
39. 기본 threshold 0.5 평가
40. ROC Curve 확인
41. F1 기준 threshold 최적화
42. 최적 threshold로 재평가
43. Logistic Regression 계수 중요도 확인
44. Permutation Importance 확인
45. joblib으로 모델 저장과 불러오기
```

---

## 5. 주요 코드 블록 설명

### 5.1 Sigmoid

```python
torch.sigmoid(x)
```

입력값을 0과 1 사이로 바꾼다.

이진 분류에서는 class 1일 확률로 해석한다.

---

### 5.2 Threshold

```python
pred = (prob >= 0.5).float()
```

확률이 0.5 이상이면 class 1, 아니면 class 0으로 예측한다.

threshold는 목적에 따라 바꿀 수 있다.

---

### 5.3 BCELoss

```python
criterion = nn.BCELoss()
loss = criterion(prob, target)
```

Sigmoid를 지난 확률값을 입력으로 받는다.

모델 구조는 보통 `Linear → Sigmoid`다.

---

### 5.4 BCEWithLogitsLoss

```python
criterion = nn.BCEWithLogitsLoss()
loss = criterion(logits, target)
```

Sigmoid 전 raw score인 logit을 입력으로 받는다.

내부에서 Sigmoid와 BCE를 안정적으로 함께 처리한다.

---

### 5.5 Iris 데이터 준비

```python
iris = load_iris()
x_data = iris.data[:100, :2]
y_data = iris.target[:100]
```

Iris 데이터에서 앞의 두 class와 앞의 두 feature만 사용한다.

이진 분류와 결정 경계 시각화를 쉽게 보기 위한 구성이다.

---

### 5.6 train_test_split

```python
train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
```

train과 validation 데이터를 나눈다.

`stratify=y`는 class 비율을 유지한다.

---

### 5.7 Tensor 변환

```python
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float().view(-1, 1)
```

입력은 float Tensor로 만들고, 정답은 BCE 손실에 맞게 `[N, 1]` 형태로 만든다.

---

### 5.8 LogisticNetWithSigmoid

```python
class LogisticNetWithSigmoid(nn.Module):
    ...
    self.l1 = nn.Linear(n_input, n_output)
    self.sigmoid = nn.Sigmoid()
```

모델 내부에 Sigmoid가 있으므로 출력이 확률값이다.

손실함수는 `BCELoss`를 사용한다.

---

### 5.9 LogisticNetLogits

```python
class LogisticNetLogits(nn.Module):
    ...
    self.l1 = nn.Linear(n_input, n_output)
```

모델 마지막에 Sigmoid가 없다.

손실함수는 `BCEWithLogitsLoss`를 사용한다.

---

### 5.10 Decision Boundary

```python
prob = torch.sigmoid(model(grid))
plt.contour(xx, yy, prob, levels=[0.5])
```

feature 공간 전체에 대해 class 1 확률을 계산하고, 확률 0.5인 지점을 선으로 그린다.

이 선이 결정 경계다.

---

### 5.11 Accuracy

```python
accuracy_score(y_true, y_pred)
```

전체 중 맞힌 비율이다.

직관적이지만 불균형 데이터에서는 위험할 수 있다.

---

### 5.12 Precision, Recall, F1

```python
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
```

- Precision은 1이라고 예측한 것 중 실제 1의 비율이다.
- Recall은 실제 1 중 모델이 찾아낸 비율이다.
- F1은 Precision과 Recall의 균형이다.

---

### 5.13 Confusion Matrix

```python
confusion_matrix(y_true, y_pred)
```

TN, FP, FN, TP를 표로 보여준다.

모델이 어떤 종류의 실수를 하는지 확인할 수 있다.

---

### 5.14 ROC-AUC

```python
roc_curve(y_true, y_score)
roc_auc_score(y_true, y_score)
```

threshold를 바꿔가며 분류 성능을 보는 지표다.

`y_score`에는 class 예측값이 아니라 확률값을 넣는다.

---

### 5.15 사기 거래 데이터

```python
make_classification(weights=[0.95, 0.05])
```

정상 거래가 95%, 사기 거래가 5%인 불균형 데이터를 만든다.

이런 데이터에서는 Recall과 F1이 중요하다.

---

### 5.16 pos_weight

```python
pos_weight = torch.tensor([normal_count / fraud_count])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

양성 class가 적을 때 해당 class를 더 중요하게 학습시킨다.

---

### 5.17 Config

```python
config = Config()
```

batch size, learning rate, hidden dimension, patience 같은 실험 설정을 한곳에 모아둔다.

---

### 5.18 DataPreprocessor

```python
preprocessor = DataPreprocessor(config)
```

데이터 불러오기, 분할, 표준화, DataLoader 생성을 하나의 클래스로 묶는다.

---

### 5.19 DiabetesClassifier

```python
Linear → BatchNorm → ReLU → Dropout
```

실무형 이진 분류 모델 구조다.

BatchNorm은 학습 안정화, Dropout은 과적합 완화에 도움을 준다.

---

### 5.20 EarlyStopping

```python
early_stopping(val_loss, model)
```

validation loss가 개선되지 않으면 학습을 멈춘다.

최적 모델 상태를 저장해둔다.

---

### 5.21 Pipeline

```python
pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(...))
])
```

전처리와 모델을 하나로 묶는다.

실무에서 모델 저장과 재사용이 쉬워진다.

---

### 5.22 ColumnTransformer

```python
ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])
```

수치형과 범주형 feature에 다른 전처리를 적용한다.

---

### 5.23 Threshold Optimization

```python
for thr in thresholds:
    y_hat = (y_proba >= thr).astype(int)
```

여러 threshold를 비교해 F1이 가장 높은 기준값을 찾는다.

---

### 5.24 Feature Importance

```python
pipe.named_steps["model"].coef_
permutation_importance(pipe, X_test, y_test, scoring="f1")
```

로지스틱 회귀 계수와 permutation importance로 feature 영향력을 해석한다.

---

### 5.25 모델 저장

```python
joblib.dump(pipe, "model.joblib")
loaded = joblib.load("model.joblib")
```

학습된 pipeline을 저장하고 다시 불러온다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `logit` | Sigmoid 전 raw score | `model(x)` 출력 |
| `prob` | 0~1 확률 | `torch.sigmoid(logit)` |
| `threshold` | class 결정 기준 | 보통 0.5 |
| `Sigmoid` | 0~1 변환 함수 | `torch.sigmoid(x)` |
| `BCE` | Binary Cross Entropy | 이진 분류 손실 |
| `BCELoss` | 확률값용 BCE | `nn.BCELoss()` |
| `BCEWithLogitsLoss` | logit용 BCE | `nn.BCEWithLogitsLoss()` |
| `Accuracy` | 전체 중 맞춘 비율 | `accuracy_score()` |
| `Precision` | 1 예측 중 실제 1 비율 | `precision_score()` |
| `Recall` | 실제 1 중 찾은 비율 | `recall_score()` |
| `F1` | Precision과 Recall 균형 | `f1_score()` |
| `Confusion Matrix` | 예측/정답 2×2 표 | `confusion_matrix()` |
| `ROC-AUC` | threshold 전체 분리 성능 | `roc_auc_score()` |
| `stratify` | class 비율 유지 분할 | `train_test_split(..., stratify=y)` |
| `pos_weight` | 양성 class 가중치 | `BCEWithLogitsLoss(pos_weight=...)` |
| `DataLoader` | mini-batch 생성 | `DataLoader(dataset, batch_size=...)` |
| `EarlyStopping` | 조기 종료 | val loss 개선 없으면 중단 |
| `Pipeline` | 전처리+모델 묶음 | scikit-learn 실무 구조 |
| `OneHotEncoder` | 범주형 인코딩 | category를 0/1 열로 변환 |
| `Permutation Importance` | 특성 중요도 | feature 섞은 뒤 성능 하락 확인 |
| `joblib` | 모델 저장 도구 | `joblib.dump`, `joblib.load` |

---

## 7. 그래프/출력 결과 해석

### 7.1 Sigmoid 그래프

입력값이 0일 때 출력이 0.5다.

입력이 커지면 1에 가까워지고, 작아지면 0에 가까워진다.

### 7.2 Iris 산점도

두 class가 feature 공간에서 어느 정도 나뉘어 있는지 확인한다.

로지스틱 회귀는 이 공간을 가르는 직선을 학습한다.

### 7.3 Loss Curve

train loss와 validation loss가 함께 감소하면 학습이 안정적이라고 볼 수 있다.

train loss만 감소하고 validation loss가 증가하면 과적합을 의심한다.

### 7.4 Accuracy Curve

accuracy가 올라가면 class 예측을 더 많이 맞히는 것이다.

train과 validation accuracy가 비슷하면 일반화가 괜찮다고 볼 수 있다.

### 7.5 Decision Boundary

확률 0.5인 지점을 선으로 그린 것이다.

이 선을 기준으로 class 0과 class 1이 나뉜다.

### 7.6 Confusion Matrix

대각선은 맞힌 개수다.

FP와 FN을 보면 모델이 어떤 실수를 하는지 확인할 수 있다.

### 7.7 ROC Curve

곡선이 왼쪽 위에 가까울수록 좋다.

AUC가 1에 가까울수록 분류 성능이 좋다.

### 7.8 Precision-Recall Curve

양성 class 탐지 품질을 볼 때 유용하다.

의료 예측이나 사기 탐지처럼 양성 class가 중요한 문제에서 특히 중요하다.

### 7.9 Threshold Optimization 그래프

threshold에 따라 precision, recall, F1이 달라진다.

문제 목적에 맞는 threshold를 선택해야 한다.

### 7.10 Feature Importance 그래프

어떤 feature가 예측에 더 영향을 주는지 확인한다.

계수 기반 중요도와 permutation importance는 서로 다른 관점의 해석이다.

---

## 8. 실습에서 배운 점

- 이진 분류는 0과 1 두 class 중 하나를 예측하는 문제다.
- Sigmoid는 logit을 확률로 바꾼다.
- 기본 threshold는 0.5다.
- BCELoss는 확률값을 입력으로 받는다.
- BCEWithLogitsLoss는 logit을 입력으로 받는다.
- BCEWithLogitsLoss가 수치적으로 더 안정적이다.
- BCEWithLogitsLoss를 사용할 때는 모델 마지막에 Sigmoid를 넣지 않는다.
- Accuracy는 직관적이지만 불균형 데이터에서는 부족하다.
- Precision, Recall, F1은 양성 class 예측 성능을 더 자세히 보여준다.
- Confusion Matrix는 모델의 실수 유형을 보여준다.
- ROC-AUC는 threshold 전체에서의 분리 성능을 본다.
- 사기 탐지처럼 불균형이 심한 문제에서는 pos_weight를 사용할 수 있다.
- 당뇨병 예측 실무형 코드는 Config, Preprocessor, Model, Trainer, EarlyStopping으로 구조화할 수 있다.
- 고객 이탈 예측처럼 수치형과 범주형이 섞이면 ColumnTransformer와 Pipeline이 유용하다.
- threshold는 0.5로 고정하지 않고 비즈니스 목적에 맞게 조정할 수 있다.
- 모델 해석에는 로지스틱 회귀 계수와 permutation importance를 함께 볼 수 있다.
- joblib으로 전처리와 모델이 묶인 pipeline을 저장할 수 있다.

---

## 9. 시험용 요약

```text
이진 분류 = 데이터를 0 또는 1 두 그룹 중 하나로 나누는 문제
```

- 회귀는 연속값 예측이고, 이진 분류는 0/1 class 예측이다.
- 이진 분류 모델은 보통 class 1일 확률을 예측한다.
- Sigmoid는 어떤 값을 0과 1 사이로 바꾼다.
- Sigmoid 입력 0의 출력은 0.5다.
- 확률이 0.5 이상이면 class 1로 예측하는 것이 기본이다.
- BCELoss는 Sigmoid를 지난 확률값을 입력으로 받는다.
- BCEWithLogitsLoss는 Sigmoid 전 logit을 입력으로 받는다.
- BCEWithLogitsLoss는 수치적으로 더 안정적이다.
- BCEWithLogitsLoss를 쓰면 모델 마지막에 Sigmoid를 붙이지 않는다.
- Accuracy는 전체 중 맞힌 비율이다.
- 불균형 데이터에서는 Accuracy만 보면 위험하다.
- Precision은 1이라고 예측한 것 중 실제 1의 비율이다.
- Recall은 실제 1 중 모델이 찾아낸 비율이다.
- F1은 Precision과 Recall의 균형이다.
- Confusion Matrix는 TN, FP, FN, TP를 보여준다.
- ROC-AUC는 threshold 전체에서의 분리 성능이다.
- stratify는 train/test class 비율을 유지하는 데 필요하다.
- StandardScaler는 train에는 fit_transform, test에는 transform만 사용한다.
- pos_weight는 소수 양성 class를 더 중요하게 학습시키는 방법이다.
- EarlyStopping은 validation loss가 좋아지지 않으면 학습을 멈춘다.
- Threshold는 0.5로 고정하지 않고 목적에 따라 조정할 수 있다.
- 고객 이탈 예측처럼 수치형과 범주형이 섞이면 ColumnTransformer와 Pipeline이 유용하다.
- 모델 저장은 joblib.dump, 불러오기는 joblib.load를 사용한다.