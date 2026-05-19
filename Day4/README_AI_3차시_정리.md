# 강의_9기_AI개론_4차시_정리
## AI 기초: 신경망, 역전파(Backprop), 활성화 함수

## 1. 개요
이번 강의는 신경망 학습의 핵심 구조를 다룬다. 핵심 흐름은 `Forward → Loss → Backward → Update`이다.

## 2. 오늘 배운 핵심 개념
- AI / ML / DL 관계
- 지도학습
- 선형 모델 `Yp = W * X + B`
- MSE 손실 함수
- 역전파 Backpropagation
- Gradient
- Learning Rate
- Optimizer
- Momentum
- 활성화 함수
- Bias-Variance
- Train / Validation / Test 분할
- Data Leakage

## 3. 실습 파일 구성
| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_4차시_01_first_ml_ok.ipynb` | 신장으로 체중 예측, 경사하강법, Optimizer |
| `강의_9기_AI개론_4차시_02_편향–분산 관찰, 데이터 분할 누수 방지_ok.ipynb` | Bias-Variance, 데이터 분할 |
| `04. AI 기초 ... .srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_4차시...pdf` | 강의 개념 자료 |

## 4. 코드 흐름 요약
```text
데이터 준비 → 전처리 → Tensor 변환 → W/B 생성
→ 예측 → 손실 계산 → 역전파 → 파라미터 업데이트
→ 반복 학습 → 그래프 확인 → Optimizer 적용
→ Bias-Variance 비교
```

## 5. 주요 코드 블록 설명
### 데이터 분리
```python
x = sampleData1[:, 0]
y = sampleData1[:, 1]
```
신장을 입력, 체중을 정답으로 분리한다.

### 전처리
```python
X = x - x.mean()
Y = y - y.mean()
```
평균을 빼서 중심을 0으로 맞춘다.

### 파라미터
```python
W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()
```
학습할 가중치와 편향을 만든다.

### 예측
```python
def pred(X):
    return W * X + B
```

### 손실 함수
```python
def mse(Yp, Y):
    return ((Yp - Y) ** 2).mean()
```

### 역전파
```python
loss.backward()
```

### 직접 업데이트
```python
with torch.no_grad():
    W -= lr * W.grad
    B -= lr * B.grad
```

### Optimizer
```python
optimizer.step()
optimizer.zero_grad()
```

## 6. 주요 함수/변수/약어 정리
| 이름 | 의미 |
|---|---|
| `X` | 입력값 |
| `Y` | 정답값 |
| `Yp` | 예측값 |
| `W` | Weight, 가중치 |
| `B` | Bias, 편향 |
| `loss` | 오차 |
| `MSE` | 평균 제곱 오차 |
| `lr` | 학습률 |
| `grad` | 기울기 |
| `epoch` | 반복 학습 단위 |
| `optimizer` | 최적화 도구 |
| `SGD` | 확률적 경사하강법 |
| `MLP` | 다층 퍼셉트론 |
| `ReLU` | 대표 활성화 함수 |

## 7. 그래프/출력 결과 해석
- 산점도는 신장과 체중의 관계를 보여준다.
- 학습 곡선에서 loss가 감소하면 학습이 진행된 것이다.
- 학습된 직선은 모델이 찾은 관계를 의미한다.
- Momentum 곡선이 더 빨리 내려가면 학습 가속 효과가 있다.
- Bias-Variance 그래프는 과소적합/과적합 여부를 확인한다.

## 8. 실습에서 배운 점
- 머신러닝의 시작은 문제 정의이다.
- Loss를 줄이는 방향으로 파라미터를 수정한다.
- 역전파는 각 파라미터의 책임을 계산한다.
- Optimizer는 업데이트 과정을 자동화한다.
- 활성화 함수는 비선형성을 추가한다.
- 데이터 분할은 데이터 누수를 막기 위해 필요하다.

## 9. 시험용 요약
- 학습 루프는 `Forward → Loss → Backward → Update`이다.
- 선형 모델은 `Yp = W * X + B`이다.
- MSE는 오차 제곱의 평균이다.
- `loss.backward()`는 gradient를 계산한다.
- `optimizer.step()`은 파라미터를 수정한다.
- `optimizer.zero_grad()`는 gradient를 초기화한다.
- 활성화 함수가 없으면 깊은 신경망도 선형 모델과 같다.
- Train은 학습, Validation은 튜닝, Test는 최종 평가에 사용한다.
