# 강의_9기_AI개론_3차시_정리  
## PyTorch 기본 기능 · Tensor 핵심 · 자동 미분

## 1. 개요

이번 3차시는 PyTorch의 기본 기능을 다룬다.

핵심은 PyTorch에서 데이터를 담는 기본 단위인 **Tensor**를 이해하고, 딥러닝 학습에 필요한 **자동 미분 Autograd**가 어떻게 작동하는지 확인하는 것이다.

2차시에서 NumPy 배열을 배웠다면, 3차시는 그 배열 사고를 PyTorch Tensor로 확장하는 단계다.

전체 흐름은 다음과 같다.

```text
NumPy ndarray
→ PyTorch Tensor
→ requires_grad
→ 계산 그래프
→ backward()
→ grad 확인
→ zero_() 초기화
```

Tensor는 NumPy 배열과 비슷해 보이지만, PyTorch Tensor에는 GPU 연산과 자동 미분 기능이 붙어 있다는 점이 다르다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| Tensor | PyTorch의 기본 숫자 컨테이너 |
| Scalar | 숫자 하나를 담는 0차원 Tensor |
| Vector | 숫자가 한 줄로 나열된 1차원 Tensor |
| Matrix | 행과 열을 가진 2차원 Tensor |
| 3D Tensor | 행렬이 여러 장 쌓인 구조 |
| 4D Tensor | 이미지 배치 데이터를 표현할 때 자주 쓰는 구조 |
| shape | Tensor의 차원별 크기 |
| dtype | Tensor 안 데이터 타입 |
| device | Tensor가 CPU/GPU 중 어디에 있는지 |
| requires_grad | 자동 미분 추적 여부 |
| Autograd | PyTorch의 자동 미분 기능 |
| backward() | 계산 그래프를 거꾸로 따라가며 gradient 계산 |
| grad | 계산된 gradient가 저장되는 속성 |
| zero_() | gradient를 0으로 초기화하는 함수 |
| view() | Tensor 모양을 바꾸는 함수 |
| item() | 원소 하나짜리 Tensor에서 Python 숫자를 꺼내는 함수 |
| torch.max() | 최댓값과 최댓값 위치를 구하는 함수 |
| Sigmoid | 입력을 0과 1 사이 값으로 바꾸는 함수 |
| Define-by-run | 실행 시점에 계산 그래프를 만드는 PyTorch 방식 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_3차시_01_pytorch_ok.ipynb` | PyTorch Tensor 생성, view, max, NumPy 변환, 자동 미분, Sigmoid gradient |
| `강의_9기_AI개론_3차시_02_기본_텐서(Tensor).ipynb` | Tensor 차원, 생성 함수, 속성, 연산, squeeze/unsqueeze, cat, broadcasting |
| `03. Pytorch 기본 _ Tensor 핵심.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_3차시_(파이토치의_기본_기능).pdf` | PyTorch 기본 기능 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. NumPy, Matplotlib, PyTorch import
2. Tensor 개념 정리
3. 0차원 Scalar Tensor 생성
4. 1차원 Vector Tensor 생성
5. 2차원 Matrix Tensor 생성
6. 3차원 Tensor 생성
7. 4차원 이미지 Tensor 구조 확인
8. zeros, ones, full, rand, randn, arange 사용
9. Tensor shape, size, dim, dtype, device, numel 확인
10. dtype 변환: float, long
11. NumPy 배열과 Tensor 변환
12. item()으로 스칼라 값 꺼내기
13. view(), reshape(), contiguous()로 모양 변경
14. squeeze(), unsqueeze(), expand(), repeat()로 차원 조정
15. chunk(), split(), cat()으로 Tensor 분할/연결
16. Tensor 기본 연산과 matmul 확인
17. Broadcasting 확인
18. Tensor slicing 확인
19. sum, max, min, torch.max(dim) 확인
20. requires_grad=True 설정
21. y = 2x² + 2 함수 계산
22. sum()으로 스칼라 생성
23. backward()로 gradient 계산
24. zero_()로 gradient 초기화
25. Sigmoid 함수와 gradient 계산
26. Define-by-run 방식 이해
```

---

## 5. 주요 코드 블록 설명

### 5.1 Tensor 생성

```python
torch.tensor([1, 2, 3])
```

값을 직접 넣어 Tensor를 만든다.

숫자 하나를 넣으면 0차원 Tensor, 리스트를 넣으면 1차원 이상 Tensor가 된다.

---

### 5.2 float 변환

```python
tensor.float()
```

Tensor를 `torch.float32` 타입으로 바꾼다.

PyTorch 모델 입력이나 weight 연산에서 float 타입을 자주 사용한다.

---

### 5.3 0차원 Tensor

```python
r0 = torch.tensor(1.0).float()
```

숫자 하나만 담고 있는 Tensor다.

Loss처럼 값 하나로 나오는 결과가 이런 형태다.

---

### 5.4 1차원 Tensor

```python
r1 = torch.tensor(np.array([1, 2, 3, 4, 5])).float()
```

NumPy 배열을 Tensor로 변환한다.

1차원 Tensor는 벡터처럼 생각하면 된다.

---

### 5.5 2차원 Tensor

```python
r2 = torch.tensor([[1, 5, 6], [4, 3, 2]]).float()
```

행과 열이 있는 행렬 Tensor다.

`shape`이 `[2, 3]`이면 2행 3열이다.

---

### 5.6 3차원 Tensor

```python
r3 = torch.randn((3, 2, 2))
```

정규분포 난수로 3차원 Tensor를 만든다.

`shape = [3, 2, 2]`는 2x2 행렬이 3장 있다는 뜻이다.

---

### 5.7 4차원 Tensor

```python
r4 = torch.ones((2, 3, 2, 2))
```

이미지 데이터 구조를 이해하는 예시다.

```text
(batch, channel, height, width)
```

PyTorch 이미지 모델에서는 보통 이 순서를 사용한다.

---

### 5.8 Tensor 속성 확인

```python
x.shape
x.size()
x.dim()
x.dtype
x.device
x.numel()
```

Tensor를 다룰 때 가장 먼저 확인해야 하는 정보다.

shape이 틀리면 모델 입력에서 대부분 에러가 난다.

---

### 5.9 dtype 변환

```python
r1.float()
r1.long()
```

- `float()`: 실수형 Tensor로 변환
- `long()`: int64 정수형 Tensor로 변환

분류 문제 label은 보통 long 타입을 사용한다.

---

### 5.10 NumPy와 Tensor 변환

```python
torch.tensor(np_array)
torch.from_numpy(np_array)
tensor.numpy()
```

NumPy 배열과 Tensor를 서로 바꿀 수 있다.

`torch.from_numpy()`는 메모리를 공유할 수 있으므로 원본 변경에 주의해야 한다.

---

### 5.11 item()

```python
r0.item()
```

원소가 하나인 Tensor에서 Python 숫자를 꺼낸다.

학습 중 `loss.item()` 형태로 자주 사용된다.

---

### 5.12 view()

```python
r3.view(3, -1)
r3.view(-1)
```

Tensor의 모양을 바꾼다.

`-1`은 해당 차원을 자동 계산하라는 뜻이다.

---

### 5.13 contiguous와 reshape

```python
y.contiguous().view(-1)
y.reshape(-1)
```

전치한 Tensor는 메모리 배치가 연속적이지 않을 수 있다.

이때 `view()`가 안 되면 `contiguous()`나 `reshape()`를 사용한다.

---

### 5.14 squeeze와 unsqueeze

```python
tensor.squeeze()
tensor.unsqueeze(0)
```

- `squeeze`: 크기 1인 차원을 제거한다.
- `unsqueeze`: 크기 1인 차원을 추가한다.

모델 입력 shape을 맞출 때 자주 사용한다.

---

### 5.15 expand와 repeat

```python
tensor.expand(4, 3, 2)
tensor.repeat(4, 1, 1)
```

- `expand`: 실제 복사 없이 확장해 보이게 한다.
- `repeat`: 실제로 데이터를 반복 복사한다.

---

### 5.16 cat

```python
torch.cat((a, b), dim=0)
torch.cat((a, b), dim=1)
```

Tensor를 연결한다.

`dim=0`은 행 방향, `dim=1`은 열 방향으로 붙인다.

---

### 5.17 torch.max

```python
torch.max(r2, 1)
torch.max(r2, 1)[1]
```

`dim`을 지정하면 최댓값과 최댓값의 위치를 함께 반환한다.

다중 분류에서는 가장 큰 점수를 가진 class 번호를 구할 때 자주 사용한다.

---

### 5.18 requires_grad

```python
x = torch.tensor(x_np, requires_grad=True, dtype=torch.float32)
```

PyTorch에게 이 Tensor와 관련된 연산을 추적하라고 설정한다.

자동 미분을 위해 가장 중요한 설정이다.

---

### 5.19 backward

```python
z.backward()
```

계산 그래프를 거꾸로 따라가며 gradient를 계산한다.

계산된 gradient는 `.grad`에 저장된다.

---

### 5.20 zero_

```python
x.grad.zero_()
```

기존 gradient를 0으로 초기화한다.

PyTorch는 gradient를 누적하므로 새 계산 전 초기화가 필요하다.

---

### 5.21 Sigmoid

```python
sigmoid = torch.nn.Sigmoid()
y = sigmoid(x)
```

입력값을 0과 1 사이로 바꾼다.

이진 분류와 기울기 소실 개념을 이해할 때 중요하다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `torch` | PyTorch 라이브러리 | `torch.tensor()`, `torch.randn()` |
| `Tensor` | PyTorch 데이터 컨테이너 | 숫자 배열 + 자동 미분 + GPU |
| `ndarray` | NumPy 배열 | Tensor와 비슷한 배열 구조 |
| `r0` | rank 0 Tensor | 0차원 스칼라 |
| `r1` | rank 1 Tensor | 1차원 벡터 |
| `r2` | rank 2 Tensor | 2차원 행렬 |
| `r3` | rank 3 Tensor | 3차원 Tensor |
| `shape` | Tensor 모양 | `tensor.shape` |
| `size()` | Tensor 모양 | `tensor.size()` |
| `dim()` | 차원 수 | `tensor.dim()` |
| `dtype` | 데이터 타입 | `tensor.dtype` |
| `device` | 저장 장치 | CPU 또는 GPU |
| `numel()` | 전체 원소 수 | `tensor.numel()` |
| `view()` | shape 변경 | `tensor.view(3, -1)` |
| `reshape()` | shape 변경 | `tensor.reshape(-1)` |
| `contiguous()` | 메모리 연속화 | 전치 후 view 전에 사용 |
| `squeeze()` | 차원 제거 | `tensor.squeeze()` |
| `unsqueeze()` | 차원 추가 | `tensor.unsqueeze(0)` |
| `expand()` | 가상 확장 | 메모리 복사 적음 |
| `repeat()` | 실제 반복 | 메모리 사용 증가 |
| `cat()` | Tensor 연결 | `torch.cat((a,b), dim=0)` |
| `max()` | 최댓값 | `torch.max(tensor, dim=1)` |
| `indices` | 최댓값 위치 | `torch.max(...)[1]` |
| `requires_grad` | 자동 미분 추적 | `requires_grad=True` |
| `grad` | gradient 저장 위치 | `x.grad` |
| `backward()` | 역전파 | `loss.backward()` |
| `zero_()` | 원본을 0으로 초기화 | `x.grad.zero_()` |
| `Sigmoid` | 0~1 변환 함수 | `torch.nn.Sigmoid()` |
| `Autograd` | 자동 미분 기능 | PyTorch가 gradient 계산 |

---

## 7. 그래프/출력 결과 해석

### 7.1 2차 함수 그래프

`y = 2x² + 2`는 U자 형태다.

x가 0일 때 가장 낮고, 양쪽으로 갈수록 값이 커진다.

### 7.2 2차 함수 gradient 그래프

미분 결과는 `y' = 4x`다.

그래프는 직선 형태이며, x가 음수면 gradient도 음수, x가 양수면 gradient도 양수다.

### 7.3 Sigmoid 그래프

Sigmoid는 S자 형태다.

입력이 작으면 0에 가깝고, 입력이 크면 1에 가까워진다.

### 7.4 Sigmoid gradient 그래프

가운데 근처에서 gradient가 가장 크고, 양 끝으로 갈수록 작아진다.

이것은 기울기 소실 문제와 연결된다.

---

## 8. 실습에서 배운 점

- Tensor는 PyTorch의 핵심 데이터 구조다.
- Tensor는 NumPy 배열과 비슷하지만 자동 미분과 GPU 연산을 지원한다.
- Tensor를 다룰 때는 shape, dtype, device를 먼저 확인해야 한다.
- 이미지 데이터는 보통 4차원 Tensor로 다룬다.
- `view()`와 `reshape()`는 Tensor 모양을 바꾸는 데 사용한다.
- `squeeze()`와 `unsqueeze()`는 차원을 줄이거나 늘릴 때 사용한다.
- `torch.max(output, 1)[1]`은 분류 모델의 예측 class를 구할 때 자주 사용된다.
- `requires_grad=True`를 설정해야 자동 미분 추적이 시작된다.
- `backward()`는 계산 그래프를 거꾸로 따라가며 gradient를 계산한다.
- gradient는 `.grad`에 저장된다.
- PyTorch gradient는 누적되므로 `zero_()`로 초기화해야 한다.
- 직접 만든 함수도 Tensor 연산으로 구성되어 있으면 자동 미분이 가능하다.
- PyTorch는 Define-by-run 방식이라 코드 실행 중 계산 그래프가 만들어진다.

---

## 9. 시험용 요약

```text
Tensor = NumPy 배열 + GPU 연산 + 자동 미분
```

- 0차원 Tensor는 스칼라다.
- 1차원 Tensor는 벡터다.
- 2차원 Tensor는 행렬이다.
- 4차원 이미지 Tensor는 `(batch, channel, height, width)` 순서로 이해한다.
- `torch.tensor()`는 값을 Tensor로 만든다.
- `.float()`는 float32 타입으로 바꾼다.
- `.long()`은 int64 타입으로 바꾼다.
- `shape`, `size()`, `dim()`, `dtype`, `device`를 확인해야 한다.
- `view()`는 Tensor 모양을 바꾼다.
- `view(-1)`은 Tensor를 한 줄로 펼친다.
- 전치 후 `view()`가 안 되면 `contiguous()`나 `reshape()`를 사용한다.
- `unsqueeze()`는 차원을 추가한다.
- `squeeze()`는 크기 1인 차원을 제거한다.
- `torch.max(output, 1)[1]`은 예측 class를 구할 때 자주 쓴다.
- `requires_grad=True`는 자동 미분 추적 시작이다.
- `backward()`는 gradient 계산이다.
- `.grad`에는 계산된 gradient가 저장된다.
- gradient는 누적되므로 `zero_()`가 필요하다.
- `y = 2x² + 2`의 미분은 `4x`다.
- Sigmoid는 0과 1 사이 값을 출력한다.
- Sigmoid의 gradient는 가운데에서 크고 양 끝에서 작다.
- PyTorch는 Define-by-run 방식이다.