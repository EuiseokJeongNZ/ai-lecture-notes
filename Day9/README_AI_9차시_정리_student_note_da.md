# 강의_9기_AI개론_9차시_정리  
## 다중 분류 · Softmax · CrossEntropyLoss · torch.max · NLLLoss · KL Divergence · 다층 신경망

## 1. 개요

이번 9차시는 다중 분류를 정리하는 강의다.

8차시에서 이진 분류를 배웠다면, 9차시는 선택지가 2개에서 여러 개로 늘어나는 구조다.

```text
이진 분류: 0 또는 1 중 하나를 고른다
다중 분류: 0, 1, 2, ..., N-1 중 하나를 고른다
```

핵심 흐름은 다음이다.

```text
다중 분류 문제 정의
→ 출력 차원이 1개에서 N개로 늘어남
→ 가중치 벡터가 가중치 행렬로 바뀜
→ Softmax로 N개 점수를 확률 분포로 변환
→ CrossEntropyLoss가 Softmax + Log + NLLLoss를 내부에서 처리
→ torch.max(outputs, 1)[1]로 예측 class 선택
→ Iris 데이터로 선형 다중 분류 실습
→ 입력 feature를 2개에서 4개로 늘려 비교
→ Wine 데이터로 다층 신경망 다중 분류 실습
→ NLLLoss와 KL Divergence 개념 정리
```

이번 강의의 핵심은 **모델은 Softmax를 붙이지 않고 logits를 그대로 출력하고, 손실함수는 `nn.CrossEntropyLoss()`를 쓰는 것**이다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| 다중 분류 | 여러 class 중 하나를 고르는 문제 |
| Logits | 모델이 직접 출력하는 raw score |
| Softmax | logits를 class별 확률 분포로 바꾸는 함수 |
| CrossEntropyLoss | PyTorch 다중 분류 표준 손실함수 |
| NLLLoss | 정답 class의 log probability만 뽑아 손실을 계산하는 함수 |
| LogSoftmax | softmax 후 log를 안정적으로 계산하는 함수 |
| Weight Matrix | class별 가중치 벡터가 모인 행렬 |
| torch.max | 최댓값과 최댓값 위치를 반환하는 함수 |
| dim=1 | `[batch, class]`에서 class 방향 |
| class index label | one-hot이 아닌 정수형 정답 label |
| Iris multi-class | 붓꽃 3개 품종 분류 실습 |
| Wine multi-class | Wine 3개 class 다층 신경망 분류 실습 |
| Dropout | 일부 뉴런을 꺼 과적합을 줄이는 기법 |
| AdamW | weight decay를 포함한 Adam 계열 Optimizer |
| KL Divergence | 두 확률분포가 얼마나 다른지 재는 값 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_9차시_01_multi_classifier_ok.ipynb` | Iris 다중 분류, CrossEntropyLoss, torch.max, Softmax 해석, NLLLoss 패턴 |
| `강의_9기_AI개론_9차시_02_다층신경망_다중분류_ok.ipynb` | Wine 데이터, 다층 신경망, Dropout, AdamW, train/val/test 평가 |
| `강의_9기_AI개론_9차시_03 초간단 NLLLoss, KL Divergence 정리_ok.ipynb` | LogSoftmax, NLLLoss, CrossEntropyLoss 관계, KL Divergence |
| `09. 다중분류 1.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_9차시_(다중분류1).pdf` | 다중 분류, Softmax, CrossEntropyLoss, NLLLoss 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. 이진 분류와 다중 분류 차이 정리
3. 다중 분류 출력 shape 확인
4. nn.Linear(2, 3)으로 가중치 행렬 확인
5. Softmax로 logits를 확률 분포로 변환
6. CrossEntropyLoss 사용법 확인
7. CrossEntropyLoss = LogSoftmax + NLLLoss 관계 확인
8. NLLLoss가 정답 class log probability만 뽑는 과정 확인
9. softmax(dim=0)과 softmax(dim=1) 차이 확인
10. torch.max(outputs, 1)[1]로 예측 class 추출
11. Iris 데이터 불러오기
12. feature 2개 선택
13. train/validation 분할
14. class별 산점도 확인
15. Tensor 변환, label은 long 타입 사용
16. MultiClassLinearNet 정의
17. CrossEntropyLoss와 SGD 설정
18. 학습 전 output shape과 초기 accuracy 확인
19. 학습 함수로 1000 epoch 학습
20. loss/accuracy curve 시각화
21. logits에 softmax를 적용해 확률 해석
22. weight matrix와 bias 확인
23. Iris feature 4개로 확장
24. 2 feature와 4 feature 성능 비교
25. Confusion Matrix와 Classification Report 확인
26. Wine 데이터 불러오기
27. train/validation/test 분할과 StandardScaler 적용
28. DeepMultiClassNet 정의
29. Dropout과 AdamW를 사용해 학습
30. Wine test 성능 평가
31. 개별 샘플 logits/probability/prediction 확인
32. 큰 logit에서 log_softmax 안정성 확인
33. CrossEntropyLoss, LogSoftmax+NLLLoss, Softmax+log+NLLLoss 패턴 비교
34. KL Divergence 개념과 예제 확인
```

---

## 5. 주요 코드 블록 설명

### 5.1 다중 분류 logits

```python
logits = torch.tensor([
    [2.1, 0.8, -1.3],
    [0.2, 2.7, 0.4],
    [-0.5, 0.3, 2.2]
])
```

각 행은 한 샘플이고, 각 열은 class별 점수다.

아직 확률이 아니라 raw score다.

---

### 5.2 nn.Linear(2, 3)

```python
linear_multi = nn.Linear(2, 3)
```

입력 feature 2개를 받아 class 3개 점수를 출력한다.

weight shape은 `[3, 2]`이고, class별 weight row가 있는 구조다.

---

### 5.3 Softmax

```python
probs = torch.softmax(logits, dim=1)
```

각 행의 class 점수를 확률 분포로 바꾼다.

`dim=1`은 class 방향이다.

---

### 5.4 CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
```

다중 분류 표준 손실함수다.

Softmax를 직접 적용하지 않고 logits를 그대로 넣는다.

---

### 5.5 LogSoftmax + NLLLoss

```python
log_probs = F.log_softmax(logits, dim=1)
loss = nn.NLLLoss()(log_probs, labels)
```

CrossEntropyLoss를 분해한 형태다.

CrossEntropyLoss와 같은 원리로 동작한다.

---

### 5.6 NLLLoss 직접 계산

```python
picked_log_probs = log_probs[torch.arange(len(labels)), labels]
manual_nll = -picked_log_probs.mean()
```

각 샘플에서 정답 class 위치의 log probability만 뽑고, 음수로 바꾼 뒤 평균낸다.

---

### 5.7 softmax dim 비교

```python
torch.softmax(outputs, dim=0)
torch.softmax(outputs, dim=1)
```

`dim=0`은 열 방향 합이 1이 되고, `dim=1`은 행 방향 합이 1이 된다.

다중 분류 예측 확률은 보통 `dim=1`이다.

---

### 5.8 torch.max

```python
values, indices = torch.max(outputs, dim=1)
pred = torch.max(outputs, 1)[1]
```

각 행에서 가장 큰 class 점수를 찾는다.

최종 예측 class는 `indices`다.

---

### 5.9 Iris 데이터 준비

```python
iris = load_iris()
x_select = iris.data[:, [0, 2]]
y_org = iris.target
```

3개 class를 모두 사용하고, 시각화를 위해 feature 2개만 선택한다.

---

### 5.10 label long 변환

```python
labels = torch.tensor(y_train).long()
```

CrossEntropyLoss의 정답 label은 반드시 정수 class index여야 한다.

그래서 dtype은 `torch.long`이다.

---

### 5.11 MultiClassLinearNet

```python
class MultiClassLinearNet(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.l1(x)
```

입력을 받아 class 개수만큼 logits를 출력하는 선형 다중 분류 모델이다.

---

### 5.12 학습 루프

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

PyTorch 다중 분류 학습의 기본 구조다.

---

### 5.13 Softmax로 모델 출력 해석

```python
sample_probs = torch.softmax(sample_logits, dim=1)
```

CrossEntropyLoss 학습 중에는 logits를 그대로 쓰지만, 결과를 사람이 해석할 때는 softmax를 적용한다.

---

### 5.14 feature 4개 확장

```python
x_all = iris.data
net4 = MultiClassLinearNet(n_input=4, n_output=3)
```

입력 feature를 2개에서 4개로 늘린다.

더 많은 정보를 쓰면 accuracy가 같더라도 loss가 낮아질 수 있다.

---

### 5.15 Wine 다층 신경망

```python
Linear(13 → 64)
→ ReLU
→ Dropout
→ Linear(64 → 32)
→ ReLU
→ Dropout
→ Linear(32 → 3)
```

Wine 데이터의 13개 feature를 입력받아 3개 class를 분류한다.

---

### 5.16 Dropout

```python
nn.Dropout(0.3)
```

학습 중 일부 뉴런 출력을 랜덤하게 꺼서 과적합을 줄인다.

---

### 5.17 AdamW

```python
optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

Adam 계열 Optimizer에 weight decay를 함께 적용한다.

---

### 5.18 큰 logit에서 안정성

```python
F.log_softmax(big_logits, dim=1)
F.cross_entropy(big_logits, target)
```

큰 logit에서도 수치적으로 안정적인 방식이다.

직접 `softmax → log`를 계산하는 방식보다 안전하다.

---

### 5.19 KL Divergence

```python
F.kl_div(torch.log(Q), P, reduction="batchmean")
```

두 확률분포 P와 Q가 얼마나 다른지 계산한다.

첫 번째 인자는 log probability여야 한다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `logits` | 모델 raw score | softmax 전 출력 |
| `Softmax` | 점수를 확률 분포로 변환 | `torch.softmax(logits, dim=1)` |
| `CrossEntropyLoss` | 다중 분류 표준 손실 | `nn.CrossEntropyLoss()` |
| `NLLLoss` | 정답 log probability에 음수 적용 | `nn.NLLLoss()` |
| `LogSoftmax` | softmax 후 log를 안정적으로 계산 | `F.log_softmax(x, dim=1)` |
| `KL Divergence` | 두 확률분포 차이 | `F.kl_div(log_q, p)` |
| `labels` | 정답 class index | dtype은 `torch.long` |
| `dim=1` | class 방향 | `[batch, class]`에서 class 축 |
| `torch.max(outputs, 1)` | 행별 최댓값과 index | 예측 class 추출 |
| `argmax(dim=1)` | 가장 큰 class index | `outputs.argmax(dim=1)` |
| `weight matrix` | class별 가중치 묶음 | `nn.Linear(in, out).weight` |
| `Dropout` | 일부 뉴런을 꺼 과적합 완화 | `nn.Dropout(0.3)` |
| `AdamW` | weight decay 포함 Adam 계열 Optimizer | `optim.AdamW(...)` |
| `StandardScaler` | 평균 0, 표준편차 1 표준화 | train에 fit, val/test에 transform |
| `Classification Report` | class별 성능 요약 | precision, recall, f1 |
| `Confusion Matrix` | class별 예측 오류 표 | `confusion_matrix()` |

---

## 7. 그래프/출력 결과 해석

### 7.1 Softmax 확률 그래프

가장 큰 logit을 가진 class의 확률이 가장 크다.

모든 class 확률의 합은 1이다.

### 7.2 Iris 산점도

class 0은 비교적 잘 분리되고, class 1과 class 2는 일부 겹칠 수 있다.

선형 모델은 직선 경계로 이 데이터를 나누려고 한다.

### 7.3 Iris Loss Curve

loss가 감소하면 모델이 정답 class에 더 높은 점수를 주도록 학습되고 있다는 뜻이다.

### 7.4 Iris Accuracy Curve

accuracy가 증가하면 예측 class가 정답과 일치하는 비율이 높아진다는 뜻이다.

### 7.5 2 Feature vs 4 Feature Loss

feature를 더 많이 사용하면 class를 더 확신 있게 예측할 수 있다.

accuracy가 비슷해도 validation loss가 낮아질 수 있다.

### 7.6 Confusion Matrix

대각선은 맞힌 개수다.

대각선 밖의 값은 어떤 class를 어떤 class로 헷갈렸는지 보여준다.

### 7.7 Wine 학습 곡선

train과 validation loss가 함께 내려가면 학습이 안정적이다.

train만 좋아지고 validation이 나빠지면 과적합을 의심한다.

### 7.8 KL Divergence 값

KL 값이 작을수록 두 확률분포가 비슷하다는 뜻이다.

---

## 8. 실습에서 배운 점

- 다중 분류는 여러 class 중 하나를 고르는 문제다.
- 이진 분류와 다르게 출력이 class 개수만큼 필요하다.
- 다중 분류의 weight는 행렬 형태다.
- 모델 출력 logits는 확률이 아니다.
- Softmax를 적용해야 확률 분포로 해석할 수 있다.
- CrossEntropyLoss는 logits를 직접 입력으로 받는다.
- CrossEntropyLoss에는 Softmax를 먼저 적용하지 않는다.
- 다중 분류 label은 one-hot이 아니라 정수 class index다.
- label dtype은 `torch.long`이어야 한다.
- 예측 class는 `torch.max(outputs, 1)[1]`로 구한다.
- NLLLoss는 정답 class의 log probability만 뽑아 손실을 계산한다.
- `LogSoftmax + NLLLoss`는 `CrossEntropyLoss`와 같은 원리다.
- `softmax(dim=1)`은 각 샘플의 class 확률을 만든다.
- feature를 늘리면 모델의 확신도가 좋아질 수 있다.
- 다층 신경망은 선형 모델보다 복잡한 패턴을 학습할 수 있다.
- Dropout은 과적합을 줄이는 데 도움을 준다.
- KL Divergence는 확률분포끼리 비교할 때 사용한다.

---

## 9. 시험용 요약

```text
다중 분류 = 여러 class 중 하나를 고르는 문제
```

- 이진 분류는 출력 1개, 다중 분류는 class 개수만큼 출력한다.
- 다중 분류 모델 출력 shape은 보통 `[batch, num_classes]`다.
- 출력 class가 N개이면 `nn.Linear(..., N)`을 사용한다.
- 이진 분류의 가중치는 벡터 느낌이고, 다중 분류의 가중치는 행렬이다.
- logits는 softmax 전 raw score다.
- Softmax는 logits를 확률 분포로 바꾼다.
- Softmax 결과는 각 행의 합이 1이다.
- `[batch, class]` 출력에서는 `softmax(dim=1)`을 사용한다.
- PyTorch 표준 패턴은 `logits + CrossEntropyLoss`다.
- `CrossEntropyLoss`에 Softmax를 먼저 적용하지 않는다.
- `CrossEntropyLoss`는 내부적으로 LogSoftmax + NLLLoss를 처리한다.
- 다중 분류 정답 label은 one-hot이 아니라 class index다.
- `CrossEntropyLoss`의 label dtype은 `torch.long`이어야 한다.
- 예측 class는 `torch.max(outputs, 1)[1]`로 구한다.
- `torch.max(outputs, 1)`은 values와 indices를 반환한다.
- 우리가 필요한 예측 label은 indices다.
- NLLLoss는 정답 class의 log probability만 뽑아 loss를 계산한다.
- `LogSoftmax + NLLLoss`는 `CrossEntropyLoss`와 같은 원리다.
- 직접 `Softmax + log`를 쓰는 방식은 수치적으로 불안정할 수 있다.
- 입력 feature를 늘리면 accuracy가 같아도 loss가 낮아질 수 있다.
- Dropout은 과적합을 줄이는 데 도움을 준다.
- KL Divergence는 두 확률분포가 얼마나 다른지 재는 값이다.