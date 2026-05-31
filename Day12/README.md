# 강의_9기_AI개론_12차시_정리  
## 심화 경사하강법과 옵티마이저 구현하기

## 1. 개요

이번 12차시는 경사하강법과 대표 Optimizer를 NumPy로 직접 구현하는 심화 실습이다.

앞 강의에서는 `optim.SGD`, `optim.Adam`, `optim.AdamW`처럼 PyTorch에 이미 구현된 Optimizer를 사용했다.  
이번 강의에서는 그 내부에서 어떤 계산이 일어나는지 직접 코드로 만든다.

핵심 흐름은 다음이다.

```text
경사하강법 복습
→ 선형회귀 y = Wx + b를 NumPy로 학습
→ cost_function으로 MSE 계산
→ gradient 함수로 dW, db 계산
→ SGD 직접 구현
→ Momentum 직접 구현
→ RMSprop 직접 구현
→ Adam 직접 구현
→ AdamW 직접 구현
→ 같은 문제에서 Optimizer별 이동 경로 비교
```

이번 강의의 핵심 질문은 다음이다.

```text
optimizer.step() 한 줄 안에서는 실제로 어떤 계산이 일어나는가?
```

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| Gradient Descent | 손실이 작아지는 방향으로 파라미터를 이동시키는 방법 |
| Cost Function | 현재 파라미터가 얼마나 틀렸는지 계산하는 함수 |
| MSE | Mean Squared Error, 평균 제곱 오차 |
| Gradient | 손실이 증가하는 방향을 나타내는 기울기 |
| Learning Rate | 한 step에서 얼마나 이동할지 정하는 값 |
| Parameter | 학습으로 바뀌는 값, 예: W, b |
| `params` | 파라미터를 담은 딕셔너리 |
| `grads` | 파라미터별 gradient를 담은 딕셔너리 |
| SGD | 가장 기본적인 Optimizer |
| Momentum | 이전 이동 방향을 기억하는 Optimizer |
| RMSprop | gradient 제곱 평균으로 파라미터별 보폭을 조절하는 Optimizer |
| Adam | Momentum과 RMSprop을 결합한 Optimizer |
| AdamW | Adam에 weight decay를 분리 적용한 Optimizer |
| Weight Decay | 파라미터 크기가 지나치게 커지는 것을 막는 규제 |
| Exponential Moving Average | 최근 값에 더 큰 비중을 두는 이동 평균 |
| Bias Correction | Adam 초반의 0 초기화 편향을 보정하는 과정 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `(심화)경사하강법과_옵티마이저_구현하기_ok.ipynb` | 원본 12강 실습 노트북 |
| `Summary_Day12_student_note_da.ipynb` | 학생 필기식 Summary 노트북 |
| `README_AI_12차시_정리_student_note_da.md` | 12차시 README 정리 파일 |

---

## 4. 코드 흐름 요약

```text
1. NumPy와 Matplotlib import
2. 예제 데이터 X, y 생성
3. W, b를 0으로 초기화
4. learning_rate와 iterations 설정
5. cost_function으로 MSE 계산
6. gradient 함수로 dW, db 계산
7. 반복문으로 기본 경사하강법 실행
8. cost history를 저장하고 그래프로 확인
9. params, grads 딕셔너리 구조 이해
10. SGD 클래스 구현
11. Momentum 클래스 구현
12. Momentum의 velocity 개념 정리
13. RMSprop 클래스 구현
14. RMSprop의 h, decay_rate, epsilon 개념 정리
15. Adam 클래스 구현
16. Adam의 m, v, m_hat, v_hat 개념 정리
17. AdamW 클래스 구현
18. weight_decay 개념 정리
19. Optimizer 비교용 2차 손실 함수 정의
20. SGD, Momentum, RMSprop, Adam, AdamW의 loss 감소 비교
21. 손실 곡면 위에서 Optimizer별 이동 경로 비교
22. 핵심 코드 패턴과 시험용 요약 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 예제 데이터

```python
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
```

입력과 정답 데이터를 만든다.

정답 관계는 `y = 2X`다.

---

### 5.2 파라미터 초기화

```python
W = 0.0
b = 0.0
```

가중치와 편향을 0으로 시작한다.

학습을 통해 `W`는 2 근처, `b`는 0 근처로 이동해야 한다.

---

### 5.3 비용 함수

```python
def cost_function(X, y, W, b):
    n = len(X)
    prediction = W * X + b
    cost = np.sum((prediction - y) ** 2) / n
    return cost
```

현재 `W`, `b`에서 예측값을 만들고 MSE를 계산한다.

---

### 5.4 기울기 함수

```python
def gradient(X, y, W, b):
    n = len(X)
    prediction = W * X + b
    dW = np.sum(2 * X * (prediction - y)) / n
    db = np.sum(2 * (prediction - y)) / n
    return dW, db
```

MSE를 `W`와 `b`에 대해 각각 미분한 값을 계산한다.

`dW`, `db`는 파라미터를 어느 방향으로 수정해야 하는지 알려준다.

---

### 5.5 기본 경사하강법

```python
dW, db = gradient(X, y, W, b)
W = W - learning_rate * dW
b = b - learning_rate * db
```

기울기의 반대 방향으로 `W`, `b`를 업데이트한다.

---

### 5.6 params와 grads

```python
params = {"W": np.array(0.0), "b": np.array(0.0)}
grads = {"W": np.array(-20.0), "b": np.array(-6.0)}
```

Optimizer를 일반화하기 위해 파라미터와 기울기를 딕셔너리로 관리한다.

---

### 5.7 SGD

```python
params[key] -= self.lr * grads[key]
```

현재 gradient만 보고 파라미터를 이동한다.

가장 기본적인 업데이트 규칙이다.

---

### 5.8 Momentum

```python
self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
params[key] += self.v[key]
```

이전 이동 방향을 velocity로 기억한다.

관성이 생겨 더 빠르게 최솟값으로 이동할 수 있다.

---

### 5.9 RMSprop

```python
self.h[key] *= self.decay_rate
self.h[key] += (1 - self.decay_rate) * (grads[key] ** 2)
params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

gradient 제곱의 이동 평균을 이용해 파라미터별 이동 크기를 조절한다.

---

### 5.10 Adam

```python
self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
m_hat = self.m[key] / (1 - self.beta1 ** self.iter)
v_hat = self.v[key] / (1 - self.beta2 ** self.iter)
params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)
```

Momentum의 방향 정보와 RMSprop의 크기 정보를 함께 사용한다.

`m_hat`, `v_hat`은 편향 보정된 값이다.

---

### 5.11 AdamW

```python
params[key] -= self.lr * self.weight_decay * params[key]
params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)
```

weight decay를 Adam update와 분리해서 적용한다.

가중치가 지나치게 커지는 것을 막아 일반화에 도움을 준다.

원본 노트북의 AdamW 마지막 줄에는 `lr_t`가 등장하지만 정의되어 있지 않아 실행 오류가 날 수 있다.  
Summary에서는 실행 가능하도록 `self.lr`로 수정했다.

---

### 5.12 Optimizer 비교 함수

```python
loss(W, b) = (W - 3)² + 0.5 × (b + 2)²
```

간단한 2차 손실 곡면을 만들고 각 Optimizer의 이동 경로를 비교한다.

최솟값은 `W = 3`, `b = -2`다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `np` | NumPy 약어 | 배열 계산 |
| `plt` | Matplotlib pyplot 약어 | 그래프 출력 |
| `X` | 입력 데이터 | 모델 입력 |
| `y` | 정답 데이터 | target |
| `W` | weight, 가중치 | 입력에 곱해지는 파라미터 |
| `b` | bias, 편향 | 예측값에 더해지는 파라미터 |
| `prediction` | 예측값 | `W * X + b` |
| `cost` | 비용 또는 손실 | 현재 오차 |
| `MSE` | Mean Squared Error | 평균 제곱 오차 |
| `gradient` | 기울기 | 손실이 커지는 방향 |
| `dW` | W에 대한 기울기 | W 업데이트에 사용 |
| `db` | b에 대한 기울기 | b 업데이트에 사용 |
| `lr` | learning rate | 이동 보폭 |
| `iterations` | 반복 횟수 | 학습 반복 수 |
| `params` | 파라미터 딕셔너리 | W, b 저장 |
| `grads` | 기울기 딕셔너리 | dW, db 저장 |
| `key` | 딕셔너리 키 | `"W"`, `"b"` |
| `val` | 딕셔너리 값 | 실제 배열 |
| `SGD` | Stochastic Gradient Descent | 기본 Optimizer |
| `Momentum` | 관성 Optimizer | velocity 사용 |
| `v` | velocity 또는 2차 모멘트 | 문맥에 따라 의미 다름 |
| `RMSprop` | RMS 기반 Optimizer | gradient 제곱 평균 사용 |
| `h` | gradient 제곱 이동 평균 | RMSprop에서 사용 |
| `decay_rate` | 감쇠율 | 이전 값을 얼마나 유지할지 정함 |
| `epsilon` | 작은 값 | 0으로 나누기 방지 |
| `Adam` | Momentum + RMSprop | m과 v 사용 |
| `m` | 1차 모멘트 | gradient 이동 평균 |
| `beta1` | m의 감쇠율 | 보통 0.9 |
| `beta2` | v의 감쇠율 | 보통 0.999 |
| `m_hat` | 편향 보정된 m | Adam update에 사용 |
| `v_hat` | 편향 보정된 v | Adam update에 사용 |
| `AdamW` | Adam + weight decay | weight decay 분리 적용 |
| `weight_decay` | 가중치 감쇠 | 파라미터 크기 규제 |

---

## 7. 그래프/출력 결과 해석

### 7.1 Cost 그래프

반복이 진행되면서 cost가 줄어들면 경사하강법이 제대로 작동하는 것이다.

### 7.2 학습된 직선 그래프

최종 직선이 데이터 점을 잘 지나가면 `W`, `b`가 적절히 학습된 것이다.

이번 데이터에서는 `W ≈ 2`, `b ≈ 0`에 가까워지는 것이 자연스럽다.

### 7.3 Optimizer Loss Comparison

SGD, Momentum, RMSprop, Adam, AdamW가 같은 2차 손실 문제에서 얼마나 빠르게 loss를 줄이는지 비교한다.

### 7.4 Optimizer Paths on Loss Surface

손실 곡면 위에서 각 Optimizer가 최솟값으로 이동하는 경로를 보여준다.

별표는 최솟값 위치다.

---

## 8. 실습에서 배운 점

- 경사하강법은 손실을 줄이는 방향으로 파라미터를 이동한다.
- Optimizer는 파라미터 업데이트 규칙이다.
- `params`와 `grads` 구조를 쓰면 여러 파라미터를 같은 방식으로 업데이트할 수 있다.
- SGD는 가장 기본적인 업데이트 방식이다.
- Momentum은 이전 이동 방향을 기억한다.
- RMSprop은 gradient 제곱 이동 평균으로 보폭을 조절한다.
- Adam은 Momentum과 RMSprop을 결합한다.
- Adam은 `m`, `v`, `m_hat`, `v_hat`을 사용한다.
- AdamW는 weight decay를 Adam update와 분리한다.
- `epsilon`은 0으로 나누는 문제를 막는다.
- learning rate는 너무 작아도 문제고 너무 커도 문제다.
- Optimizer별 이동 경로를 보면 업데이트 방식 차이가 더 잘 보인다.

---

## 9. 시험용 요약

```text
Optimizer = gradient를 이용해 parameter를 어떻게 업데이트할지 정하는 규칙
```

- 경사하강법 기본 식은 `param = param - lr × grad`다.
- `W`는 weight, `b`는 bias다.
- MSE는 평균 제곱 오차다.
- `cost_function`은 현재 파라미터의 손실을 계산한다.
- `gradient`는 `dW`, `db`를 계산한다.
- `learning_rate`는 한 step 이동 크기다.
- `params`는 파라미터 딕셔너리다.
- `grads`는 기울기 딕셔너리다.
- SGD는 현재 gradient만 사용한다.
- Momentum은 velocity로 이전 이동 방향을 기억한다.
- RMSprop은 gradient 제곱 이동 평균으로 보폭을 조절한다.
- Adam은 Momentum과 RMSprop을 결합한다.
- Adam의 `m`은 gradient 이동 평균이다.
- Adam의 `v`는 gradient 제곱 이동 평균이다.
- `m_hat`, `v_hat`은 편향 보정된 값이다.
- AdamW는 Adam에 weight decay를 분리 적용한다.
- `weight_decay`는 파라미터가 지나치게 커지는 것을 막는다.
- `epsilon`은 0으로 나누는 것을 막는 작은 값이다.