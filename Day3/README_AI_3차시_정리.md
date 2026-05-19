# 강의_9기_AI개론_3차시_정리  
## PyTorch 기본 기능 · Tensor 핵심

## 1. 개요

이번 강의는 PyTorch의 기본 기능을 학습하는 시간이다.

핵심은 PyTorch에서 데이터를 다루는 기본 단위인 **Tensor**를 이해하고, 딥러닝 학습에서 중요한 **자동 미분 Autograd**가 어떻게 동작하는지 코드 흐름으로 익히는 것이다.

강의 자료에서는 PyTorch Tensor를 단순한 숫자 배열이 아니라, 자동 미분과 GPU 가속을 지원하는 딥러닝용 데이터 구조로 설명한다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 설명 |
|---|---|
| Tensor | PyTorch의 기본 데이터 구조 |
| Scalar | 0차원 Tensor, 숫자 하나 |
| Vector | 1차원 Tensor |
| Matrix | 2차원 Tensor |
| 3D Tensor | 여러 행렬이 쌓인 구조 |
| 4D Tensor | 이미지 배치 데이터 표현에 자주 사용 |
| dtype | Tensor 데이터 타입 |
| shape | Tensor의 차원별 크기 |
| device | Tensor가 저장된 장치, CPU 또는 GPU |
| requires_grad | 자동 미분 추적 여부 |
| Autograd | PyTorch의 자동 미분 기능 |
| backward() | 역전파를 실행하여 gradient 계산 |
| grad | 계산된 미분값 |
| zero_() | gradient 초기화 |
| view() | Tensor 형태 변환 |
| item() | Tensor에서 Python 숫자 추출 |
| max() | 최댓값과 인덱스 확인 |
| Sigmoid | 0과 1 사이 값으로 변환하는 함수 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_3차시_01_pytorch_ok.ipynb` | PyTorch Tensor 생성, 형태 변환, 자동 미분, Sigmoid 경사 계산 |
| `03. Pytorch 기본 _ Tensor 핵심.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_3차시_(파이토치의_기본_기능).pdf` | 강의 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. NumPy, Matplotlib, PyTorch import
2. 0차원 Tensor 생성
3. 1차원 Tensor 생성
4. 2차원 Tensor 생성
5. 3차원 Tensor 생성
6. 4차원 Tensor 생성
7. Tensor dtype 변환
8. view()로 Tensor 형태 변환
9. item()으로 스칼라 값 추출
10. max()로 최댓값과 인덱스 확인
11. Tensor를 NumPy로 변환
12. requires_grad=True 설정
13. y = 2x² + 2 함수 계산
14. sum()으로 스칼라 생성
15. backward()로 gradient 계산
16. gradient 그래프 확인
17. zero_()로 gradient 초기화
18. Sigmoid 함수 경사 계산
19. 직접 만든 Sigmoid 함수의 gradient 계산
```

---

## 5. 주요 코드 블록 설명

### 5.1 PyTorch import

```python
import torch
```

PyTorch 라이브러리를 불러온다.

---

### 5.2 0차원 Tensor

```python
r0 = torch.tensor(1.0).float()
```

숫자 하나를 Tensor로 만든다.

`float()`는 데이터 타입을 `float32`로 맞춘다.

---

### 5.3 1차원 Tensor

```python
r1_np = np.array([1, 2, 3, 4, 5])
r1 = torch.tensor(r1_np).float()
```

NumPy 배열을 PyTorch Tensor로 변환한다.

---

### 5.4 2차원 Tensor

```python
r2_np = np.array([[1, 5, 6], [4, 3, 2]])
r2 = torch.tensor(r2_np).float()
```

2행 3열 Tensor를 만든다.

---

### 5.5 3차원 Tensor

```python
torch.manual_seed(123)
r3 = torch.randn((3, 2, 2))
```

정규분포 난수로 3차원 Tensor를 만든다.

---

### 5.6 4차원 Tensor

```python
r4 = torch.ones((2, 3, 2, 2))
```

4차원 Tensor를 만든다.

이미지 데이터에서는 보통 `(batch, channel, height, width)` 구조를 사용한다.

---

### 5.7 dtype 변환

```python
r5 = r1.long()
```

Tensor를 정수형 `int64`로 변환한다.

분류 문제의 라벨에서 자주 사용된다.

---

### 5.8 view()

```python
r6 = r3.view(3, -1)
r7 = r3.view(-1)
```

Tensor의 형태를 바꾼다.

`-1`은 해당 차원 크기를 자동 계산하라는 의미이다.

---

### 5.9 requires_grad

```python
x = torch.tensor(x_np, requires_grad=True, dtype=torch.float32)
```

자동 미분을 위해 Tensor 연산 추적을 시작한다.

---

### 5.10 2차 함수 계산

```python
y = 2 * x**2 + 2
```

Tensor 연산으로 2차 함수를 계산한다.

---

### 5.11 sum()

```python
z = y.sum()
```

`backward()`를 호출하기 위해 벡터 Tensor를 스칼라 Tensor로 만든다.

---

### 5.12 backward()

```python
z.backward()
```

계산 그래프를 거꾸로 따라가며 미분값을 계산한다.

---

### 5.13 grad

```python
print(x.grad)
```

계산된 gradient를 확인한다.

`y = 2x² + 2`의 미분은 `y' = 4x`이므로 `x.grad`는 `4x`와 같다.

---

### 5.14 zero_()

```python
x.grad.zero_()
```

누적된 gradient를 0으로 초기화한다.

PyTorch는 `backward()`를 호출할 때마다 gradient를 누적하기 때문에 반드시 초기화가 필요하다.

---

### 5.15 Sigmoid

```python
sigmoid = torch.nn.Sigmoid()
y = sigmoid(x)
```

PyTorch 내장 Sigmoid 함수를 사용한다.

---

### 5.16 직접 만든 Sigmoid

```python
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))
```

Tensor 연산으로 직접 Sigmoid 함수를 구현한다.

직접 만든 함수도 Tensor 연산으로 구성되어 있으면 자동 미분이 가능하다.

---

## 6. 주요 함수 / 변수 / 약어 정리

| 이름 | 의미 | 설명 |
|---|---|---|
| `torch` | PyTorch | 딥러닝 라이브러리 |
| `Tensor` | Tensor | PyTorch 데이터 구조 |
| `r0` | rank 0 | 0차원 스칼라 Tensor |
| `r1` | rank 1 | 1차원 벡터 Tensor |
| `r2` | rank 2 | 2차원 행렬 Tensor |
| `r3` | rank 3 | 3차원 Tensor |
| `r4` | rank 4 | 4차원 Tensor |
| `dtype` | data type | 데이터 타입 |
| `shape` | shape | Tensor 모양 |
| `device` | device | CPU/GPU 저장 위치 |
| `requires_grad` | gradient tracking | 자동 미분 추적 여부 |
| `grad` | gradient | 미분값 |
| `backward()` | backward pass | 역전파 실행 |
| `zero_()` | in-place zero | gradient 초기화 |
| `view()` | reshape | Tensor 형태 변경 |
| `item()` | scalar extraction | Python 숫자 추출 |
| `max()` | maximum | 최댓값 찾기 |
| `indices` | index | 최댓값 위치 |
| `Sigmoid` | sigmoid function | 0~1 사이 값 출력 |
| `Autograd` | automatic differentiation | 자동 미분 |

---

## 7. 그래프 / 출력 결과 해석

### 7.1 `y = 2x² + 2` 그래프

아래로 볼록한 U자 형태의 2차 함수이다.

x가 0일 때 y가 가장 작고, x가 양쪽으로 멀어질수록 y가 커진다.

### 7.2 Gradient 그래프

`y = 2x² + 2`의 미분은 `y' = 4x`이다.

따라서 gradient 그래프는 직선 형태이다.

- x가 음수이면 gradient도 음수
- x가 양수이면 gradient도 양수
- x가 0이면 gradient는 0

### 7.3 Sigmoid 그래프

Sigmoid는 S자 곡선이다.

- x가 작으면 0에 가까움
- x가 크면 1에 가까움
- x가 0 근처일 때 가장 빠르게 변함

### 7.4 Sigmoid gradient 그래프

Sigmoid의 gradient는 x=0 근처에서 가장 크고, 양 끝으로 갈수록 작아진다.

이 특징은 딥러닝에서 기울기 소실 문제와 연결된다.

---

## 8. 실습에서 배운 점

- PyTorch Tensor는 딥러닝에서 데이터를 표현하는 기본 단위이다.
- Tensor는 NumPy 배열과 비슷하지만 자동 미분과 GPU 연산을 지원한다.
- Tensor의 `shape`, `dtype`, `device`를 확인하는 습관이 중요하다.
- `view()`는 Tensor 모양을 바꾸는 데 사용된다.
- `item()`은 스칼라 Tensor에서 Python 숫자를 꺼낼 때 사용한다.
- `torch.max(output, 1)[1]`은 분류 모델의 예측 라벨을 구할 때 자주 사용된다.
- `requires_grad=True`를 설정해야 Autograd가 연산을 추적한다.
- `backward()`는 계산 그래프를 거꾸로 따라가며 gradient를 계산한다.
- PyTorch는 gradient를 누적하므로 `zero_()`로 초기화해야 한다.
- Sigmoid 함수도 자동 미분으로 gradient를 계산할 수 있다.
- Tensor 연산으로 직접 만든 함수도 Autograd가 처리할 수 있다.

---

## 9. 시험용 요약

- Tensor는 PyTorch의 기본 데이터 구조이다.
- 0차원은 스칼라, 1차원은 벡터, 2차원은 행렬이다.
- 이미지 데이터는 보통 4차원 Tensor로 표현한다.
- 4차원 이미지 Tensor 구조는 `(batch, channel, height, width)`이다.
- `float()`는 실수형 Tensor로 변환한다.
- `long()`은 정수형 Tensor로 변환한다.
- `shape`은 Tensor의 모양을 나타낸다.
- `device`는 Tensor가 CPU/GPU 중 어디에 있는지 나타낸다.
- `requires_grad=True`는 자동 미분 추적을 시작한다.
- `view()`는 Tensor의 형태를 바꾼다.
- `view(-1)`은 Tensor를 1차원으로 펼친다.
- `item()`은 원소 하나짜리 Tensor에서 Python 숫자를 꺼낸다.
- `max()`는 최댓값과 그 위치를 구한다.
- `backward()`는 역전파로 gradient를 계산한다.
- `.grad`에는 계산된 gradient가 저장된다.
- `zero_()`는 gradient를 0으로 초기화한다.
- `y = 2x² + 2`의 미분은 `y' = 4x`이다.
- Sigmoid 함수는 0과 1 사이 값을 출력한다.
- Sigmoid의 gradient는 가운데에서 크고 양 끝에서 작다.