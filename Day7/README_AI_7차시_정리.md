# 강의_9기_AI개론_7차시_정리  
## 선형회귀 · nn.Linear · 머신러닝/딥러닝 기본 이론

## 1. 개요

이번 강의는 **선형회귀 Linear Regression**를 중심으로 PyTorch의 가장 기본 Layer인 `nn.Linear`를 학습한다.

선형회귀는 입력값과 출력값 사이의 선형 관계를 가정하고, 가장 적절한 직선을 찾는 회귀 모델이다.

핵심 수식은 다음과 같다.

```text
y = Wx + b
```

PyTorch에서는 이 선형 함수를 `nn.Linear`로 구현한다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 설명 |
|---|---|
| 회귀 Regression | 연속적인 숫자를 예측하는 문제 |
| 분류 Classification | 정해진 클래스 중 하나를 예측하는 문제 |
| 선형 함수 | `y = Wx + b` 형태의 함수 |
| `nn.Linear` | PyTorch의 선형 변환 Layer |
| weight | 입력값의 영향력을 나타내는 가중치 |
| bias | 기본 보정값, 절편 |
| OLS | 오차 제곱합을 최소화하는 방식 |
| `nn.Module` | PyTorch 모델 정의의 부모 클래스 |
| `__init__` | 모델의 Layer를 선언하는 곳 |
| `forward()` | 데이터가 흐르는 순서를 정의하는 곳 |
| `MSELoss` | 회귀 문제의 대표 손실 함수 |
| Gradient Descent | 손실을 줄이는 방향으로 파라미터를 수정하는 방법 |
| R² Score | 모델이 데이터 분산을 얼마나 설명하는지 나타내는 평가 지표 |
| Ridge | L2 정규화 |
| Lasso | L1 정규화 |
| ElasticNet | L1과 L2를 결합한 정규화 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_7차시_01_regression_ok.ipynb` | `nn.Linear`, 선형회귀, MSELoss, 학습 루프, 다중 회귀 |
| `강의_9기_AI개론_7차시_02_기본_딥러닝 및 머신러닝 이론.ipynb` | AI/ML/DL 기본 이론, Scikit-Learn, PyTorch 분류 예제 |
| `7차시.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_7차시_(선형회귀).pdf` | 선형회귀 개념 자료 |
| `제목 없는 문서.docx` | 강의 이론 정리 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. nn.Linear(2, 3) 생성
3. 1입력 1출력 선형 함수 생성
4. weight, bias 확인
5. weight=2, bias=1로 초기화
6. y = 2x + 1 테스트
7. 2입력 1출력 함수 테스트
8. 2입력 3출력 함수 테스트
9. nn.Module을 상속한 Net 클래스 정의
10. 모델 인스턴스 생성 및 예측
11. MSELoss로 손실 계산
12. loss.backward()로 gradient 계산
13. 회귀 데이터 준비
14. 선형회귀 모델 학습
15. 학습 곡선 출력
16. 회귀 직선 시각화
17. R² Score 계산
18. 2입력 다중 회귀로 확장
19. 학습률 발산 문제 확인
20. Scikit-Learn과 PyTorch 예제 비교
21. Ridge, Lasso, ElasticNet 정규화 개념 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 `nn.Linear`

```python
l = nn.Linear(2, 3)
```

입력 2개를 받아 출력 3개를 만드는 선형 함수이다.

---

### 5.2 1입력 1출력

```python
l1 = nn.Linear(1, 1)
```

가장 기본적인 `y = wx + b` 형태이다.

---

### 5.3 파라미터 확인

```python
for name, param in l1.named_parameters():
    print(name, param)
```

Layer 내부의 weight와 bias를 확인한다.

---

### 5.4 파라미터 초기화

```python
nn.init.constant_(l1.weight, 2.0)
nn.init.constant_(l1.bias, 1.0)
```

weight를 2, bias를 1로 설정하여 `y = 2x + 1`을 만든다.

---

### 5.5 입력 shape 변경

```python
x = x.view(-1, 1)
```

`nn.Linear`에 넣기 위해 `[N]` 형태를 `[N, 1]`로 바꾼다.

---

### 5.6 커스텀 모델 클래스

```python
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.l1(x)
```

PyTorch 모델의 기본 구조이다.

---

### 5.7 MSELoss

```python
criterion = nn.MSELoss()
loss = criterion(outputs, labels)
```

예측값과 정답의 평균 제곱 오차를 계산한다.

---

### 5.8 학습 루프

```python
optimizer.zero_grad()
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

PyTorch 학습의 핵심 패턴이다.

---

### 5.9 R² Score

```python
r2 = 1 - (ss_res / ss_tot)
```

모델이 전체 데이터 변동성 중 얼마나 설명했는지를 나타낸다.

---

### 5.10 Ridge / Lasso

```python
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-2)
```

`weight_decay`는 Ridge, 즉 L2 정규화에 해당한다.

Lasso는 L1 penalty를 loss에 직접 더해 구현한다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 의미 |
|---|---|
| `x` | 입력값 |
| `y` | 정답값 |
| `W` | Weight, 가중치 |
| `b` | Bias, 편향 |
| `nn.Linear` | 선형 변환 Layer |
| `in_features` | 입력 특성 수 |
| `out_features` | 출력 특성 수 |
| `nn.Module` | PyTorch 모델 부모 클래스 |
| `__init__` | Layer 선언 위치 |
| `forward` | 순전파 정의 |
| `MSELoss` | 평균 제곱 오차 |
| `optimizer` | 파라미터 수정 도구 |
| `SGD` | 경사하강법 계열 Optimizer |
| `lr` | Learning Rate, 학습률 |
| `R²` | 결정계수 |
| `Ridge` | L2 정규화 |
| `Lasso` | L1 정규화 |
| `ElasticNet` | L1 + L2 정규화 |

---

## 7. 그래프/출력 결과 해석

### 7.1 산점도

입력값과 정답값의 관계를 확인한다.

점들이 직선에 가까운 흐름을 보이면 선형회귀가 적합할 수 있다.

### 7.2 학습 곡선

Loss가 반복 횟수에 따라 줄어들면 학습이 정상적으로 진행된 것이다.

### 7.3 회귀 직선

학습된 weight와 bias로 만든 직선이다.

직선이 점들의 흐름을 잘 따라가면 좋은 모델이다.

### 7.4 학습률 발산

학습률이 너무 크면 loss가 커지거나 `nan`이 될 수 있다.

---

## 8. 실습에서 배운 점

- 선형회귀는 딥러닝의 가장 기본 블록이다.
- `nn.Linear`는 `y = xWᵀ + b`를 계산한다.
- `weight`와 `bias`는 학습을 통해 바뀐다.
- PyTorch 모델은 `nn.Module`을 상속해서 만든다.
- `__init__`은 Layer 선언, `forward`는 데이터 흐름 정의이다.
- `net(inputs)`는 내부적으로 `forward(inputs)`를 실행한다.
- 회귀 문제에는 MSELoss를 주로 사용한다.
- 학습률은 너무 크면 발산하고 너무 작으면 학습이 느리다.
- R² Score는 모델의 설명력을 보여준다.
- Ridge, Lasso, ElasticNet은 과적합을 줄이기 위한 정규화 기법이다.

---

## 9. 시험용 요약

```text
선형회귀 = y = Wx + b
```

- 회귀는 연속적인 숫자를 예측한다.
- 분류는 정해진 클래스 중 하나를 예측한다.
- `nn.Linear(in_features, out_features)`는 선형 변환이다.
- weight shape은 `[out_features, in_features]`이다.
- bias shape은 `[out_features]`이다.
- `nn.Module`을 상속해 커스텀 모델을 만든다.
- `__init__`에서는 Layer를 선언한다.
- `forward()`에서는 데이터 흐름을 정의한다.
- `MSELoss`는 회귀 문제의 대표 손실 함수이다.
- 학습 루프는 예측 → 손실 계산 → 역전파 → 업데이트이다.
- `optimizer.zero_grad()`는 gradient 초기화이다.
- `loss.backward()`는 gradient 계산이다.
- `optimizer.step()`은 파라미터 수정이다.
- 학습률이 너무 크면 발산할 수 있다.
- R² Score는 모델의 설명력을 나타낸다.
- Ridge는 L2 정규화, Lasso는 L1 정규화이다.
- ElasticNet은 L1과 L2를 결합한 정규화이다.