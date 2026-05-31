# 강의_9기_AI개론_2차시_정리  
## NumPy · Matplotlib · 행렬 연산 기초

## 1. 개요

이번 2차시는 NumPy와 Matplotlib의 기초를 정리하는 강의다.

핵심은 NumPy 문법을 무작정 외우는 것이 아니라, 데이터를 하나씩 for문으로 처리하는 방식에서 벗어나 **배열 전체를 한 번에 처리하는 사고방식**을 익히는 것이다.

강의에서 가장 중요한 흐름은 다음과 같다.

```text
Python List → NumPy Array → PyTorch Tensor
```

Python List는 범용적이지만 수치 연산에는 느릴 수 있다.  
NumPy Array는 연속된 메모리 구조와 벡터화 연산을 통해 대량의 숫자 계산을 빠르게 처리한다.  
PyTorch Tensor는 여기에 GPU 가속과 자동 미분 기능이 더해진 구조라고 이해하면 된다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| NumPy | 숫자 배열을 빠르게 계산하기 위한 라이브러리 |
| ndarray | NumPy의 다차원 배열 컨테이너 |
| Vectorization | 반복문 없이 배열 전체를 한 번에 계산하는 방식 |
| Broadcasting | shape이 다른 배열을 자동 확장해서 계산하는 기능 |
| Shape | 배열의 모양 |
| ndim | 배열의 차원 수 |
| dtype | 배열 원소의 데이터 타입 |
| Boolean Indexing | 조건에 맞는 데이터만 고르는 방법 |
| reshape | 원소 개수는 유지하고 배열 모양만 바꾸는 함수 |
| transpose | 축 순서를 바꾸는 기능 |
| axis | 집계나 연결 방향을 정하는 축 |
| Aggregation | 여러 값을 하나의 값이나 작은 차원으로 요약하는 연산 |
| Matplotlib | 데이터를 그래프로 시각화하는 라이브러리 |
| Line Plot | 값의 변화 흐름을 선으로 보는 그래프 |
| Scatter Plot | 두 변수의 관계를 점으로 보는 그래프 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_2차시_01_numpy_ok.ipynb` | NumPy 배열 생성, shape, indexing, reshape, transpose, broadcasting, 집계 함수 |
| `강의_9기_AI개론_2차시_02_matplotlib_ok.ipynb` | Matplotlib 기본 설정, 산점도, sigmoid 그래프, 여러 선 겹쳐 그리기 |
| `강의_9기_AI개론_2차시_03_Numpy_Matplotlib_완성코드.ipynb` | 배열 생성, dtype, broadcasting, axis, 조건 인덱싱, 그래프 실습 완성 코드 |
| `강의_9기_AI개론_2차시_04_BigdataCert_Numpy.ipynb` | 빅데이터 실기 스타일 NumPy 연습, reshape, indexing, where, sort, argsort |
| `02. Numpy·Matplotlib, 행렬 연산 기초.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_2차시_(Numpy, Matplotlib 기초프로그래밍).pdf` | 강의 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. NumPy와 Matplotlib import
2. Python List와 NumPy Array 차이 확인
3. 1차원 배열과 2차원 배열 생성
4. shape, ndim, dtype 확인
5. zeros, ones, randn, linspace, arange 사용
6. indexing과 slicing 연습
7. Boolean Indexing으로 조건 필터링
8. np.where로 조건에 따른 값 변경
9. reshape로 배열 모양 변경
10. transpose로 행/열 또는 축 순서 변경
11. vstack, hstack, concatenate로 배열 연결
12. element-wise 연산과 broadcasting 확인
13. sum, mean, max, min, std, var 집계 함수 사용
14. axis=0, axis=1 의미 정리
15. argmax, argsort, sort 차이 확인
16. 정확도 계산
17. Min-Max 정규화 구현
18. Matplotlib으로 line plot, scatter plot, sigmoid graph 작성
19. 그래프 제목, 축 라벨, 범례 추가
20. 학습 곡선 형태로 train/validation loss 해석
```

---

## 5. 주요 코드 블록 설명

### 5.1 NumPy import

```python
import numpy as np
```

NumPy를 `np`라는 이름으로 불러온다.

`np.array()`, `np.arange()`, `np.mean()`처럼 사용한다.

---

### 5.2 배열 생성

```python
n1 = np.array([1, 2, 3, 4, 5, 6])
```

Python list를 NumPy 배열로 바꾼다.

NumPy 배열은 수치 계산에 적합하다.

---

### 5.3 shape, ndim, dtype 확인

```python
n1.shape
n1.ndim
n1.dtype
```

- `shape`: 배열의 모양
- `ndim`: 차원 수
- `dtype`: 데이터 타입

배열을 다룰 때 가장 먼저 확인해야 하는 정보다.

---

### 5.4 zeros와 ones

```python
np.zeros(5)
np.ones((2, 3))
```

0 또는 1로 채워진 배열을 만든다.

초기값, 마스크, 더미 데이터 생성에 사용한다.

---

### 5.5 random.randn

```python
np.random.randn(2, 3, 4)
```

정규분포 난수로 배열을 만든다.

괄호 안 숫자는 shape이다.

---

### 5.6 linspace와 arange

```python
np.linspace(-1, 1, 11)
np.arange(-1, 1.2, 0.2)
```

- `linspace`: 시작과 끝 사이를 지정한 개수만큼 균등하게 나눈다.
- `arange`: 시작부터 끝 전까지 step 간격으로 만든다.

그래프 x축 값 생성에 자주 사용한다.

---

### 5.7 인덱싱과 슬라이싱

```python
n2[:, 0]
n2[1, :]
```

- `:`는 전체를 의미한다.
- `n2[:, 0]`은 모든 행의 첫 번째 열을 가져온다.
- `n2[1, :]`은 두 번째 행 전체를 가져온다.

---

### 5.8 Boolean Indexing

```python
mask = data > 0
filtered = data[mask]
```

조건을 만족하는 데이터만 가져온다.

이상치 제거, 특정 class 선택, 결측치 처리 같은 전처리에 자주 사용한다.

---

### 5.9 np.where

```python
np.where(arr < 0, 0, arr)
```

조건이 참이면 0, 거짓이면 원래 값을 반환한다.

조건별 값 치환에 사용한다.

---

### 5.10 reshape

```python
n9.reshape(3, -1)
```

원소 개수는 유지하고 배열 모양만 바꾼다.

`-1`은 NumPy가 해당 차원을 자동 계산하라는 뜻이다.

---

### 5.11 transpose

```python
n2.T
np.transpose(n10, (1, 2, 0))
```

2차원에서는 행과 열을 바꾸고, 3차원 이상에서는 축 순서를 바꾼다.

---

### 5.12 배열 연결

```python
np.vstack([n13, n14])
np.hstack([n13, n14])
np.concatenate([n13, n14], axis=1)
```

배열을 세로 또는 가로 방향으로 붙인다.

`axis`를 정확히 이해해야 한다.

---

### 5.13 브로드캐스팅

```python
batch_data + bias
```

shape이 다른 배열을 자동 확장해서 계산한다.

딥러닝에서 batch 데이터에 bias를 더하는 구조와 연결된다.

---

### 5.14 집계 함수

```python
np.sum(n1)
np.mean(n1)
np.max(n1)
np.min(n1)
np.std(n1)
np.var(n1)
```

여러 값을 하나의 숫자 또는 작은 차원으로 요약한다.

손실 함수 Loss도 결국 예측값과 정답 차이를 하나의 숫자로 만드는 집계 함수로 볼 수 있다.

---

### 5.15 axis

```python
np.mean(scores, axis=0)
np.mean(scores, axis=1)
```

2차원 기준으로:

```text
axis=0: 세로 방향으로 내려가며 계산
axis=1: 가로 방향으로 옆으로 계산
```

---

### 5.16 argmax와 argsort

```python
arr.argmax()
np.argsort(arr)
```

- `argmax`: 가장 큰 값의 index를 반환한다.
- `argsort`: 정렬했을 때의 index 순서를 반환한다.

다중 분류에서 가장 큰 점수의 class를 찾을 때 `argmax`를 많이 쓴다.

---

### 5.17 정확도 계산

```python
matched = (yt == yp)
correct = matched.sum()
accuracy = correct / len(matched)
```

예측과 정답이 일치하는지 비교하고, True 개수를 전체 개수로 나눈다.

---

### 5.18 Min-Max 정규화

```python
normalized = (n1 - n1.min()) / (n1.max() - n1.min())
```

모든 값을 0과 1 사이로 바꾼다.

입력 feature 범위를 맞출 때 사용한다.

---

### 5.19 Matplotlib plot

```python
plt.plot(x, y, label="sin(x)")
plt.title("Line Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```

선 그래프를 그린다.

제목, 축 라벨, 범례는 그래프 해석에 필요하다.

---

### 5.20 scatter

```python
plt.scatter(xs, ys)
```

두 변수의 관계를 점으로 보여준다.

상관관계, 군집, 이상치, 분리 가능성을 확인할 때 사용한다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `np` | NumPy 별명 | `np.array()`, `np.arange()` |
| `plt` | pyplot 별명 | `plt.plot()`, `plt.scatter()` |
| `array` | 배열 | 같은 타입 데이터를 담는 구조 |
| `ndarray` | NumPy 다차원 배열 | `np.array()` 결과 |
| `shape` | 배열 모양 | `arr.shape` |
| `ndim` | 차원 수 | `arr.ndim` |
| `dtype` | 데이터 타입 | `arr.dtype` |
| `axis` | 축 | `axis=0`, `axis=1` |
| `mask` | 조건 배열 | `data > 0` |
| `reshape` | 모양 변경 | `arr.reshape(3, -1)` |
| `T` | 전치 | `arr.T` |
| `vstack` | 세로 연결 | `np.vstack([a, b])` |
| `hstack` | 가로 연결 | `np.hstack([a, b])` |
| `concatenate` | 배열 연결 | `np.concatenate([a, b], axis=0)` |
| `broadcasting` | 자동 확장 연산 | `(3,3) + (3,)` |
| `sum` | 합계 | `np.sum(arr)` |
| `mean` | 평균 | `np.mean(arr)` |
| `std` | 표준편차 | `np.std(arr)` |
| `var` | 분산 | `np.var(arr)` |
| `argmax` | 최댓값 위치 | `arr.argmax()` |
| `argsort` | 정렬 인덱스 | `np.argsort(arr)` |
| `linspace` | 균등 분할 | `np.linspace(0, 1, 11)` |
| `arange` | 간격 배열 | `np.arange(0, 10, 2)` |
| `scatter` | 산점도 | `plt.scatter(x, y)` |

---

## 7. 그래프/출력 결과 해석

### 7.1 Line Plot

선 그래프는 값이 어떻게 증가하거나 감소하는지 흐름을 보여준다.

함수 그래프나 학습 곡선에 자주 사용된다.

### 7.2 Scatter Plot

산점도는 두 변수 간 관계를 보여준다.

확인할 수 있는 것은 다음과 같다.

- 상관관계
- 군집
- 이상치
- 분리 가능성

### 7.3 Sigmoid Graph

Sigmoid는 S자 모양이다.

기울기 조절값 `a`가 커지면 더 급하게 변한다.

### 7.4 Train/Validation Loss Graph

Train Loss와 Validation Loss를 함께 그리면 과적합 여부를 확인할 수 있다.

Train Loss는 계속 줄어드는데 Validation Loss가 다시 올라가면 과적합을 의심한다.

---

## 8. 실습에서 배운 점

- NumPy는 딥러닝 데이터 처리의 기초다.
- Python List는 범용적이지만 대량 수치 연산에는 느리다.
- NumPy는 배열 전체를 한 번에 계산하는 벡터화 연산을 지원한다.
- `shape`, `axis`, `dtype`은 배열을 다룰 때 가장 먼저 확인해야 한다.
- Boolean Indexing은 조건에 맞는 데이터를 골라내는 전처리 핵심 도구다.
- Broadcasting은 반복문 없이 batch 데이터에 bias를 더하는 구조와 연결된다.
- 집계 함수는 Loss 계산과 연결된다.
- Matplotlib은 단순히 그래프를 예쁘게 그리는 도구가 아니라, 데이터와 모델 상태를 진단하는 도구다.
- 그래프에는 제목, 축 라벨, 범례가 있어야 해석하기 쉽다.

---

## 9. 시험용 요약

```text
NumPy = 배열 단위 사고 + 벡터화 연산
Matplotlib = 데이터 분포와 학습 상태를 눈으로 확인하는 도구
```

- NumPy 배열은 PyTorch Tensor의 기초다.
- Python List는 수치 연산보다 범용 데이터 저장에 가깝다.
- NumPy는 반복문 없이 배열 전체를 한 번에 계산한다.
- `shape`은 배열의 모양이다.
- `ndim`은 차원 수다.
- `dtype`은 데이터 타입이다.
- `reshape()`는 원소 개수는 유지하고 모양만 바꾼다.
- `-1`은 자동 계산 차원이다.
- `.T`는 2차원 배열에서 행과 열을 바꾼다.
- Boolean Indexing은 조건 필터링이다.
- `np.where()`는 조건에 따른 값 변경이다.
- Broadcasting은 shape이 다른 배열을 자동 확장해서 계산한다.
- `axis=0`은 세로 방향, `axis=1`은 가로 방향이다.
- 집계 함수는 데이터를 요약한다.
- Loss도 예측값과 정답 차이를 하나의 숫자로 만드는 집계 함수다.
- `argmax`는 가장 큰 값의 위치를 반환한다.
- 산점도는 상관관계, 군집, 이상치, 분리 가능성을 확인한다.
- 좋은 그래프에는 제목, 축 라벨, 범례가 필요하다.