# 강의_9기_AI개론_4차시_정리  
## AI 기초 · 신경망 · 역전파 · 활성화 함수

## 1. 개요

이번 4차시는 신경망 학습의 핵심 흐름을 실제 코드로 정리하는 강의다.

앞 차시에서 Tensor와 자동 미분을 배웠다면, 이번 차시는 그 기능을 이용해 실제 머신러닝 학습 루프를 구현한다.

핵심 흐름은 다음이다.

```text
문제 정의 → 데이터 준비 → 예측 계산 → 손실 계산 → 경사 계산 → 파라미터 수정 → 반복 → 평가
```

PyTorch 학습 코드로 보면 다음 구조가 계속 반복된다.

```text
Forward → Loss → Backward → Update
```

이번 실습의 중심 문제는 신장으로 체중을 예측하는 선형회귀다.  
5명의 신장과 체중 데이터를 가지고, 이 관계를 가장 잘 설명하는 직선을 찾는다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| AI | 사람처럼 판단하거나 행동하는 넓은 기술 |
| ML | 데이터로부터 규칙을 학습하는 AI |
| DL | 여러 층의 신경망으로 복잡한 패턴을 학습하는 ML |
| 지도학습 | 입력과 정답이 함께 있는 데이터로 학습 |
| 선형 모델 | `Yp = W * X + B` 형태의 가장 기본 모델 |
| 문제 정의 | 입력, 정답, 목표, 평가 지표를 정하는 단계 |
| 경사하강법 | Loss가 줄어드는 방향으로 파라미터를 조금씩 수정하는 방법 |
| 역전파 | 출력의 오류를 입력 방향으로 되돌리며 각 파라미터의 gradient를 계산하는 과정 |
| MSE | 예측값과 정답 차이를 제곱한 뒤 평균낸 손실 함수 |
| requires_grad | PyTorch가 gradient를 추적하게 하는 설정 |
| torch.no_grad | gradient 추적을 잠시 끄는 블록 |
| Optimizer | 파라미터 업데이트를 대신 수행하는 도구 |
| SGD | 기본 경사하강법 계열 Optimizer |
| Momentum | 이전 이동 방향의 관성을 반영하는 옵션 |
| 활성화 함수 | 선형 모델에 비선형성을 추가하는 함수 |
| Bias-Variance | 모델의 단순함과 복잡함 사이의 균형 문제 |
| Data Leakage | 평가 데이터가 학습 과정에 섞이는 문제 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_4차시_01_first_ml_ok.ipynb` | 신장/체중 선형회귀, MSE, backward, 직접 업데이트, Optimizer, Momentum |
| `강의_9기_AI개론_4차시_02_편향–분산 관찰, 데이터 분할 누수 방지_ok.ipynb` | Bias-Variance 실습, 사인파 데이터, train/validation/test 분할 |
| `04. AI 기초 (신경망, 역전파(Backprop), 활성화함수).srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_4차시_AI 기초_신경망, 역전파(Backprop), 활성화함수.pdf` | 강의 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. AI / ML / DL 관계 정리
3. 학습 방식 네 가지 정리
4. 선형회귀 문제 정의
5. 경사하강법 흐름 정리
6. 손실 지형 위에서 gradient descent 이동 시뮬레이션
7. 신장/체중 데이터 생성
8. x와 y 분리
9. 산점도로 데이터 확인
10. 평균을 빼서 데이터 중심을 0으로 이동
11. NumPy 배열을 Tensor로 변환
12. W, B를 requires_grad=True로 생성
13. pred 함수로 예측값 계산
14. mse 함수로 손실 계산
15. loss.backward()로 W.grad, B.grad 계산
16. in-place 업데이트 오류 확인
17. torch.no_grad() 안에서 직접 파라미터 수정
18. 500 epoch 반복 학습 루프 작성
19. history에 loss 기록
20. 학습 곡선 시각화
21. 학습된 직선 시각화
22. optim.SGD로 Optimizer 사용
23. optimizer.step(), optimizer.zero_grad()로 업데이트 자동화
24. momentum=0.9 적용 후 SGD와 비교
25. 국소 최적해 개념 확인
26. 활성화 함수 필요성 정리
27. Sigmoid, Tanh, ReLU 그래프 비교
28. nn.Sequential로 MLP 구조 확인
29. Bias-Variance 실습용 사인파 데이터 생성
30. Train/Validation/Test 데이터 분할
31. 작은 모델과 큰 모델 비교
32. Train/Validation loss 그래프 해석
33. Test MSE로 최종 평가
```

---

## 5. 주요 코드 블록 설명

### 5.1 선형 모델

```python
def linear_model(X, W, B):
    return W * X + B
```

입력 `X`에 기울기 `W`를 곱하고 절편 `B`를 더한다.

선형회귀의 가장 기본 형태다.

---

### 5.2 데이터 생성

```python
sampleData1 = np.array([
    [166, 58.7],
    [176.0, 75.7],
    [171.0, 62.1],
    [173.0, 70.4],
    [169.0, 60.1]
])
```

5명의 신장과 체중 데이터를 2차원 배열로 만든다.

각 행은 한 사람이고, 0번째 열은 신장, 1번째 열은 체중이다.

---

### 5.3 입력과 정답 분리

```python
x = sampleData1[:, 0]
y = sampleData1[:, 1]
```

`:`는 모든 행을 의미한다.

`0`번 열은 신장이고, `1`번 열은 체중이다.

---

### 5.4 평균 빼기 전처리

```python
X = x - x.mean()
Y = y - y.mean()
```

데이터 중심을 0으로 옮긴다.

값의 절대 크기를 줄이면 경사하강법이 더 안정적으로 작동할 수 있다.

---

### 5.5 Tensor 변환

```python
X = torch.tensor(X).float()
Y = torch.tensor(Y).float()
```

NumPy 배열을 PyTorch Tensor로 바꾼다.

`.float()`는 float32 타입으로 변환한다.

---

### 5.6 파라미터 W, B 생성

```python
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()
```

`W`와 `B`는 학습 대상 파라미터다.

`requires_grad=True`를 설정해야 `loss.backward()` 후 `.grad`가 계산된다.

---

### 5.7 예측 함수

```python
def pred(X):
    return W * X + B
```

현재 W와 B로 예측값을 계산한다.

이 함수가 이번 실습에서는 모델 역할을 한다.

---

### 5.8 MSE 손실 함수

```python
def mse(Yp, Y):
    loss = ((Yp - Y) ** 2).mean()
    return loss
```

예측값과 정답의 차이를 제곱하고 평균을 낸다.

회귀 문제에서 자주 사용하는 손실 함수다.

---

### 5.9 역전파

```python
loss.backward()
```

계산 그래프를 거꾸로 따라가며 W와 B의 gradient를 계산한다.

계산된 값은 `W.grad`, `B.grad`에 저장된다.

---

### 5.10 잘못된 업데이트

```python
W -= lr * W.grad
B -= lr * B.grad
```

`requires_grad=True`인 leaf Tensor를 직접 in-place 수정하면 에러가 난다.

파라미터 직접 수정은 `torch.no_grad()` 안에서 해야 한다.

---

### 5.11 올바른 직접 업데이트

```python
with torch.no_grad():
    W -= lr * W.grad
    B -= lr * B.grad

W.grad.zero_()
B.grad.zero_()
```

`torch.no_grad()` 안에서는 계산 그래프가 만들어지지 않는다.

업데이트 후에는 gradient를 0으로 초기화한다.

---

### 5.12 반복 학습 루프

```python
for epoch in range(num_epochs):
    Yp = pred(X)
    loss = mse(Yp, Y)
    loss.backward()

    with torch.no_grad():
        W -= lr * W.grad
        B -= lr * B.grad

    W.grad.zero_()
    B.grad.zero_()
```

예측, 손실, 경사, 수정, 초기화가 반복된다.

PyTorch 학습 루프의 가장 기본 구조다.

---

### 5.13 history 기록

```python
item = np.array([epoch, loss.item()])
history = np.vstack((history, item))
```

epoch와 loss 값을 배열에 기록한다.

나중에 학습 곡선을 그리기 위해 필요하다.

---

### 5.14 Optimizer 사용

```python
optimizer = optim.SGD([W, B], lr=lr)
```

W와 B를 Optimizer에게 맡긴다.

직접 `W -= lr * W.grad`를 쓰지 않아도 된다.

---

### 5.15 Optimizer 학습 루프

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

- `loss.backward()`: gradient 계산
- `optimizer.step()`: 파라미터 수정
- `optimizer.zero_grad()`: gradient 초기화

이 순서가 중요하다.

---

### 5.16 Momentum

```python
optimizer = optim.SGD([W, B], lr=lr, momentum=0.9)
```

이전 이동 방향의 관성을 반영한다.

수렴을 빠르게 하거나 진동을 줄이는 데 도움을 줄 수 있다.

---

### 5.17 활성화 함수

```python
torch.sigmoid(x)
torch.tanh(x)
torch.relu(x)
```

- Sigmoid는 0과 1 사이로 출력한다.
- Tanh는 -1과 1 사이로 출력한다.
- ReLU는 음수는 0, 양수는 그대로 둔다.

선형 함수만 여러 층 쌓으면 여전히 선형이므로, 활성화 함수가 비선형성을 추가한다.

---

### 5.18 MLP 구조

```python
nn.Sequential(
    nn.Linear(1, hidden),
    nn.ReLU(),
    nn.Linear(hidden, hidden),
    nn.ReLU(),
    nn.Linear(hidden, 1)
)
```

Linear와 Activation을 반복해서 만든 신경망 구조다.

모든 최신 모델도 이런 기본 블록을 확장한 형태로 볼 수 있다.

---

### 5.19 train_test_split

```python
train_test_split(X, y, test_size=0.4, random_state=42)
```

데이터를 train/validation/test로 나눌 때 사용한다.

Test 데이터는 최종 평가용으로만 사용해야 한다.

---

### 5.20 model.train(), model.eval()

```python
model.train()
model.eval()
```

- `model.train()`: 학습 모드다.
- `model.eval()`: 평가 모드다.

Dropout, BatchNorm 같은 Layer가 있을 때 특히 중요하다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `X` | 입력 데이터 | 모델에 넣는 값 |
| `Y` | 정답 데이터 | 모델이 맞혀야 하는 값 |
| `Yp` | Prediction | 예측값 |
| `W` | Weight | 선형 모델의 기울기 |
| `B` | Bias | 선형 모델의 절편 |
| `loss` | 손실 | 예측이 얼마나 틀렸는지 |
| `grad` | Gradient | 파라미터 수정 방향 |
| `lr` | Learning Rate | 한 번에 이동하는 크기 |
| `epoch` | 반복 단위 | 학습 반복 횟수 |
| `requires_grad` | 자동 미분 추적 | `requires_grad=True` |
| `backward()` | 역전파 | `loss.backward()` |
| `zero_()` | gradient 초기화 | `W.grad.zero_()` |
| `torch.no_grad()` | gradient 추적 중지 | 직접 업데이트나 평가에 사용 |
| `optim.SGD` | SGD Optimizer | `optim.SGD(params, lr=...)` |
| `optimizer.step()` | 파라미터 수정 | gradient 기준 업데이트 |
| `optimizer.zero_grad()` | gradient 초기화 | Optimizer가 관리하는 grad 초기화 |
| `momentum` | 관성 옵션 | 이전 이동 방향 반영 |
| `MSE` | 평균 제곱 오차 | `((Yp - Y)**2).mean()` |
| `nn.Linear` | 선형 Layer | `nn.Linear(in, out)` |
| `nn.ReLU` | 활성화 함수 | 음수는 0, 양수는 그대로 |
| `nn.Sequential` | Layer 묶음 | 여러 Layer를 순서대로 연결 |
| `model.train()` | 학습 모드 | 학습 시 호출 |
| `model.eval()` | 평가 모드 | 검증/테스트 시 호출 |
| `train_test_split` | 데이터 분할 | train/validation/test 분할 |

---

## 7. 그래프/출력 결과 해석

### 7.1 신장/체중 산점도

점 하나가 한 사람의 신장과 체중이다.

점들이 오른쪽 위로 올라가는 경향이 있으면 신장이 클수록 체중이 증가하는 경향이 있다고 볼 수 있다.

### 7.2 평균 제거 후 산점도

데이터의 중심이 0 근처로 이동한다.

상대적인 관계는 유지되지만, 경사하강법이 더 안정적으로 작동하기 쉬운 형태가 된다.

### 7.3 학습 곡선

Loss가 epoch가 증가할수록 줄어들면 학습이 진행되고 있다는 뜻이다.

초반에 빠르게 줄고 뒤로 갈수록 완만해지는 형태가 자주 나온다.

### 7.4 학습된 직선

학습된 W와 B로 만든 직선이다.

점들의 흐름을 잘 대표하면 모델이 관계를 어느 정도 학습한 것이다.

### 7.5 SGD vs Momentum 그래프

Momentum을 적용한 곡선이 더 빠르게 내려가면 학습 가속 효과가 있다고 볼 수 있다.

### 7.6 활성화 함수 그래프

Sigmoid, Tanh, ReLU는 서로 다른 모양을 가진다.

ReLU는 양수 영역에서 gradient가 유지되어 딥러닝에서 자주 쓰인다.

### 7.7 Bias-Variance 그래프

Train loss와 Validation loss를 함께 봐야 한다.

Train loss는 낮은데 Validation loss가 높으면 과적합을 의심한다.

---

## 8. 실습에서 배운 점

- 머신러닝의 첫 단계는 문제 정의다.
- 입력 X와 정답 Y를 명확히 정해야 한다.
- 선형회귀는 딥러닝의 가장 기본 단위다.
- 경사하강법은 Loss를 줄이는 방향으로 조금씩 이동한다.
- `loss.backward()`가 역전파를 수행한다.
- `requires_grad=True`가 있어야 gradient가 계산된다.
- 직접 파라미터를 수정할 때는 `torch.no_grad()`가 필요하다.
- gradient는 누적되므로 매 반복마다 초기화해야 한다.
- Optimizer를 쓰면 파라미터 업데이트가 간결해진다.
- `optimizer.step()`은 업데이트, `optimizer.zero_grad()`는 초기화다.
- Momentum은 학습을 더 빠르고 안정적으로 만들 수 있다.
- 활성화 함수는 신경망에 비선형성을 추가한다.
- 선형 함수만 쌓으면 깊어져도 선형 모델과 같아진다.
- Bias-Variance는 모델이 너무 단순한지, 너무 복잡한지 판단하는 기준이다.
- Train/Validation/Test 분할은 데이터 누수를 막기 위해 필요하다.

---

## 9. 시험용 요약

```text
PyTorch 학습 루프 = Forward → Loss → Backward → Update
```

- 모든 딥러닝의 시작은 선형 모델이다.
- 선형회귀 기본식은 `Yp = W * X + B`다.
- 문제 정의가 입력, 정답, 손실 함수, 평가 지표를 결정한다.
- 경사하강법은 Loss가 줄어드는 방향으로 파라미터를 수정한다.
- Learning Rate는 한 번에 이동하는 크기다.
- MSE는 오차 제곱의 평균이다.
- `requires_grad=True`는 gradient 추적 시작이다.
- `loss.backward()`는 gradient 계산이다.
- `.grad`에는 계산된 gradient가 저장된다.
- 직접 파라미터 수정은 `torch.no_grad()` 안에서 한다.
- gradient는 누적되므로 초기화해야 한다.
- `optimizer.step()`은 파라미터 업데이트다.
- `optimizer.zero_grad()`는 gradient 초기화다.
- Momentum은 이전 이동 방향의 관성을 반영한다.
- 선형 함수만 여러 층 쌓아도 여전히 선형이다.
- 활성화 함수는 비선형성을 추가한다.
- ReLU는 대표적인 활성화 함수다.
- High Bias는 과소적합과 연결된다.
- High Variance는 과적합과 연결된다.
- Train은 학습, Validation은 튜닝, Test는 최종 평가다.
- Test 데이터가 학습에 섞이면 데이터 누수다.