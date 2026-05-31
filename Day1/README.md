# 강의_9기_AI개론_1차시_정리  
## 딥러닝 필수 파이썬 · 실습 환경구성 · OOP

## 1. 개요

이번 1차시는 딥러닝을 바로 구현하기 전에, 앞으로 PyTorch 코드를 읽기 위해 필요한 파이썬 기본 개념을 정리하는 시간이다.

강의에서 제일 먼저 잡아야 하는 흐름은 이것이다.

```text
Forward → Loss → Backward → Update
```

사람이 문제를 풀고, 채점하고, 틀린 이유를 확인하고, 다시 고치는 것처럼 딥러닝도 예측하고, 손실을 계산하고, 역전파로 원인을 찾고, 파라미터를 수정한다.

이번 차시는 복잡한 수식을 외우는 것보다 “딥러닝 구조가 파이썬 코드로 어떻게 보이는지”를 이해하는 데 목적이 있다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 정리 |
|---|---|
| Forward | 입력 데이터를 모델에 넣어 예측값을 만드는 과정 |
| Loss | 예측값과 정답을 비교해서 얼마나 틀렸는지 숫자로 표현 |
| Backward | Loss에서 시작해 거꾸로 기울기를 계산하는 과정 |
| Optimizer | 계산된 gradient를 이용해 parameter를 수정하는 도구 |
| Learning Rate | 한 번에 얼마나 움직일지 정하는 값 |
| Container | list, dict, NumPy array처럼 데이터를 담는 그릇 |
| Reference | 변수명이 실제 데이터가 아니라 메모리 위치를 가리키는 것 |
| copy() | 원본과 독립적인 복사본을 만드는 함수 |
| Composite Function | 여러 함수가 순서대로 연결된 구조 |
| Numerical Differentiation | 아주 작은 차이를 이용해 기울기를 근사하는 방식 |
| Autograd | PyTorch의 자동 미분 기능 |
| Class | 객체를 만들기 위한 설계도 |
| Instance | 클래스로 만든 실제 객체 |
| Attribute | 객체가 가진 데이터 |
| Method | 객체가 가진 함수 |
| __call__ | 객체를 함수처럼 호출하게 해주는 메서드 |
| nn.Module | PyTorch 모델을 만들 때 상속하는 기본 클래스 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_1차시_01_intro_ok.ipynb` | 실습 환경, PyTorch 학습 흐름, 이미지 분류 예고 코드 |
| `강의_9기_AI개론_1차시_02_python_ok.ipynb` | NumPy, Tensor, copy, 함수, 미분, Class, 상속, __call__ |
| `강의_9기_AI개론_1차시_03_python.ipynb` | Python 기본 자료형, list, tuple, dict, 반복문, 함수 |
| `01. 딥러닝 필수 파이썬, 실습 환경구성 및 OOP.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_1차시_(딥러닝에_필요한_파이썬의_개념).pdf` | 개념 정리 PDF |

---

## 4. 코드 흐름 요약

```text
1. NumPy, Matplotlib, PyTorch import
2. 딥러닝 학습 흐름 정리
3. Loss와 parameter update 공식 확인
4. Python 기본 자료형 확인
5. list, tuple, dict 사용법 확인
6. for문, if문, 함수 정의 복습
7. list와 NumPy array 연산 차이 확인
8. NumPy array의 참조 복사 문제 확인
9. copy()로 안전한 복사 확인
10. Tensor와 NumPy 변환 시 메모리 공유 확인
11. 2차 함수 정의와 그래프 출력
12. 합성 함수 구현
13. 중앙 차분 수치 미분 구현
14. Sigmoid 함수와 미분 확인
15. PyTorch Autograd로 자동 미분 확인
16. Class, 상속, __call__ 실습
17. nn.Module 기반 PyTorch 모델 구조 예고
18. 간단한 PyTorch 학습 루프 형태 확인
```

---

## 5. 주요 코드 블록 설명

### 5.1 라이브러리 import

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
```

`np`, `plt`는 별명이다.  
NumPy는 수치 계산, Matplotlib은 그래프, PyTorch는 Tensor와 자동 미분에 사용한다.

---

### 5.2 Loss 예시

```python
target = 5
prediction = 4
loss = abs(target - prediction)
```

정답과 예측값의 차이를 손실로 생각하는 가장 단순한 예시다.

실제 딥러닝에서는 MSELoss나 CrossEntropyLoss 같은 손실 함수를 사용한다.

---

### 5.3 Parameter Update

```python
new_param = param - (lr * grad)
```

gradient 방향의 반대 방향으로 parameter를 수정한다.

`lr`이 너무 크면 발산하고, 너무 작으면 학습이 느리다.

---

### 5.4 list와 NumPy array 차이

```python
[1, 2, 3] + [4, 5, 6]
np.array([1, 2, 3]) + np.array([4, 5, 6])
```

list는 서로 이어 붙고, NumPy array는 같은 위치의 값끼리 더해진다.

딥러닝에서는 숫자 배열 연산이 중요하므로 NumPy/Tensor 방식에 익숙해져야 한다.

---

### 5.5 얕은 복사 문제

```python
x = np.array([5, 7, 9])
y = x
x[1] = -1
```

`y = x`는 값 전체를 새로 복사하는 것이 아니라 같은 데이터를 같이 바라보는 참조 복사다.

그래서 x를 바꾸면 y도 같이 바뀐다.

---

### 5.6 copy()

```python
y = x.copy()
```

원본과 독립적인 복사본을 만든다.

원본 데이터를 보존해야 할 때 꼭 사용한다.

---

### 5.7 함수 정의

```python
def square(x):
    return x * x
```

함수는 입력을 받아 결과를 반환하는 코드 묶음이다.

`print()`는 보여주는 것이고, `return`은 실제 결과를 밖으로 돌려주는 것이다.

---

### 5.8 합성 함수

```python
y = f3(f2(f1(x)))
```

딥러닝 모델도 여러 함수가 순서대로 연결된 거대한 합성 함수로 이해할 수 있다.

---

### 5.9 수치 미분

```python
def fdiff(f):
    def diff(x):
        h = 1e-6
        return (f(x+h) - f(x-h)) / (2*h)
    return diff
```

중앙 차분으로 기울기를 근사한다.

이 코드는 미분 원리 이해용이고, 실제 학습에서는 PyTorch Autograd를 사용한다.

---

### 5.10 PyTorch Autograd

```python
x = torch.tensor(3.0, requires_grad=True)
y = 2 * x**2 + 2
y.backward()
print(x.grad)
```

`requires_grad=True`는 PyTorch가 계산 과정을 추적하라는 뜻이다.

`backward()`를 호출하면 gradient가 계산되고, `.grad`에 저장된다.

---

### 5.11 Class

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

Class는 데이터와 기능을 묶는 설계도이다.

`__init__`은 객체를 만들 때 실행되는 초기 설정 함수다.

---

### 5.12 상속과 super()

```python
class Circle(Point):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.r = r
```

부모 클래스의 기능을 물려받고, 필요한 속성을 추가한다.

PyTorch 모델에서도 `super().__init__()`이 자주 나온다.

---

### 5.13 __call__

```python
class H:
    def __call__(self, x):
        return 2 * x**2 + 2

h = H()
h(3)
```

`__call__`이 있으면 객체를 함수처럼 호출할 수 있다.

PyTorch에서 `model(x)`가 가능한 이유도 이 구조와 연결된다.

---

### 5.14 nn.Module

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)
```

PyTorch 모델의 기본 구조다.

`__init__`에서는 Layer를 준비하고, `forward()`에서는 데이터 흐름을 정의한다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `np` | NumPy 별명 | `np.array()`, `np.arange()` |
| `plt` | Matplotlib pyplot | `plt.plot()`, `plt.show()` |
| `torch` | PyTorch | `torch.tensor()`, `torch.ones()` |
| `nn` | Neural Network 모듈 | `nn.Linear()`, `nn.MSELoss()` |
| `optim` | Optimizer 모듈 | `optim.SGD()` |
| `x` | 입력값 | 함수나 모델에 넣는 값 |
| `y` | 정답 또는 출력값 | 예측해야 하는 값 |
| `pred` | Prediction | 모델 예측값 |
| `loss` | 손실 | 예측이 얼마나 틀렸는지 |
| `grad` | Gradient | 수정 방향 |
| `lr` | Learning Rate | 한 번에 움직이는 크기 |
| `epoch` | 반복 학습 단위 | 전체 데이터를 한 번 학습 |
| `forward` | 순전파 | 입력에서 예측으로 가는 과정 |
| `backward` | 역전파 | Loss에서 거꾸로 gradient 계산 |
| `copy()` | 복사 함수 | 원본 보호용 복사 |
| `__init__` | 생성자 | 객체 초기 설정 |
| `__call__` | 호출 메서드 | 객체를 함수처럼 호출 |
| `self` | 자기 자신 | 객체 내부 속성/메서드 접근 |
| `requires_grad` | 자동 미분 추적 | `torch.tensor(..., requires_grad=True)` |
| `zero_()` | 원본을 0으로 수정 | gradient 초기화 |

---

## 7. 그래프/출력 결과 해석

### 7.1 2차 함수 그래프

`f(x) = 2x² + 2`는 U자 모양이다.

x가 0 근처일 때 가장 낮고, 양쪽으로 갈수록 값이 커진다.

### 7.2 함수와 도함수 그래프

원래 함수는 곡선이고, 도함수는 기울기를 나타낸다.

기울기가 음수이면 왼쪽에서 내려가는 중이고, 양수이면 오른쪽에서 올라가는 중이다.

### 7.3 Sigmoid 그래프

Sigmoid는 입력값을 0과 1 사이로 바꾼다.

x가 0일 때 출력은 0.5다.

### 7.4 Sigmoid 미분 그래프

가운데 근처에서 기울기가 크고, 양 끝으로 갈수록 작아진다.

이 특성은 나중에 기울기 소실 문제와 연결된다.

### 7.5 Point / Circle 그래프

Class를 사용해 점과 원을 객체로 표현했다.

데이터와 기능을 하나의 객체로 묶는 구조를 눈으로 확인한 예제다.

---

## 8. 실습에서 배운 점

- 딥러닝은 예측, 채점, 역추적, 수정의 반복이다.
- PyTorch는 자동 미분과 Optimizer를 제공해서 수학 구현 부담을 줄여준다.
- list와 NumPy array는 연산 방식이 다르다.
- 컨테이너 변수는 참조 문제 때문에 원본이 같이 바뀔 수 있다.
- 원본 보존이 필요하면 `copy()`를 사용해야 한다.
- 함수는 입력을 출력으로 바꾸는 구조이고, 딥러닝 모델도 거대한 함수처럼 볼 수 있다.
- 수치 미분은 미분 원리를 이해하기 위한 방법이다.
- 실제 딥러닝에서는 PyTorch Autograd가 gradient를 자동 계산한다.
- Class는 데이터와 기능을 묶는 구조다.
- `__call__` 구조를 알면 `model(x)`가 왜 되는지 이해할 수 있다.
- PyTorch 모델은 `nn.Module`을 상속하고, `__init__`과 `forward()`를 사용한다.

---

## 9. 시험용 요약

```text
딥러닝 학습 = Forward → Loss → Backward → Update
```

- Loss는 모델이 얼마나 틀렸는지를 나타낸다.
- Gradient는 어느 방향으로 고쳐야 하는지를 알려준다.
- Optimizer는 gradient를 이용해 parameter를 수정한다.
- 학습률이 너무 크면 발산하고, 너무 작으면 느리다.
- `y = x`는 같은 데이터를 참조할 수 있다.
- 원본을 보호하려면 `copy()`를 사용한다.
- 딥러닝 모델은 여러 함수가 연결된 합성 함수다.
- 수치 미분은 원리 이해용이다.
- PyTorch는 Autograd로 미분을 자동 처리한다.
- `requires_grad=True`는 미분 추적 시작이다.
- `backward()`는 gradient 계산이다.
- Class는 객체 설계도다.
- `__call__` 덕분에 객체를 함수처럼 부를 수 있다.
- PyTorch 모델은 `nn.Module`을 상속해서 만든다.
- `__init__`은 Layer 준비, `forward()`는 데이터 흐름 정의다.