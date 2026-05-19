# 강의_9기_AI개론_2차시_정리  
## Numpy / Matplotlib 기초프로그래밍

## 1. 개요

이번 강의는 NumPy와 Matplotlib의 기초를 학습하는 시간이다.

핵심은 단순히 문법을 외우는 것이 아니라, 데이터를 리스트처럼 하나씩 처리하는 방식에서 벗어나 **배열 단위로 사고하는 법**을 익히는 것이다.

강의 자료에서는 오늘의 목표를 다음 두 가지로 정리한다.

1. NumPy 배열과 연산
2. 데이터 해석과 시각화

NumPy는 PyTorch Tensor의 기초가 되며, Matplotlib은 데이터 분포와 모델 학습 상태를 눈으로 확인하는 진단 도구이다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 설명 |
|---|---|
| NumPy | 고속 수치 연산을 위한 Python 라이브러리 |
| ndarray | NumPy의 다차원 배열 객체 |
| Vectorization | 반복문 없이 배열 전체를 한 번에 계산하는 방식 |
| Broadcasting | shape가 다른 배열을 자동 확장하여 연산하는 기능 |
| Axis | 다차원 배열에서 연산 방향을 결정하는 축 |
| Aggregation | 여러 값을 하나의 값 또는 더 작은 차원으로 요약하는 연산 |
| Boolean Indexing | 조건에 맞는 데이터만 선택하는 필터링 방법 |
| Matplotlib | 데이터를 그래프로 시각화하는 라이브러리 |
| Scatter Plot | 두 변수의 관계를 점으로 보여주는 그래프 |
| Line Plot | 값의 변화 흐름을 선으로 보여주는 그래프 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_2차시_01_numpy_ok.ipynb` | NumPy 배열 생성, shape, indexing, reshape, transpose, broadcasting, aggregation |
| `강의_9기_AI개론_2차시_02_matplotlib_ok.ipynb` | Matplotlib 기본 설정, 산점도, 함수 그래프, subplot |
| `02. Numpy·Matplotlib, 행렬 연산 기초.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_2차시_(Numpy, Matplotlib 기초프로그래밍).pdf` | 강의 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. NumPy import
2. 1차원 / 2차원 배열 생성
3. shape, ndim, dtype 확인
4. zeros, ones, randn 배열 생성
5. linspace, arange로 수치 배열 생성
6. indexing, slicing, boolean indexing
7. reshape로 배열 모양 변경
8. transpose로 축 교체
9. vstack, hstack, concatenate로 배열 연결
10. 배열끼리 element-wise 연산
11. broadcasting으로 shape가 다른 배열 연산
12. universal function 사용
13. sum, mean, max, min, std, var 집계
14. 정확도 계산
15. min-max normalization
16. Matplotlib으로 line plot, scatter plot, sigmoid graph 작성
17. subplot으로 이미지 여러 개 출력
```

---

## 5. 주요 코드 블록 설명

### 5.1 NumPy import

```python
import numpy as np
```

NumPy를 `np`라는 이름으로 불러온다.

---

### 5.2 배열 생성

```python
n1 = np.array([1, 2, 3, 4, 5, 6])
```

Python 리스트를 NumPy 배열로 변환한다.

---

### 5.3 shape 확인

```python
n1.shape
```

배열의 모양을 확인한다.

---

### 5.4 2차원 배열

```python
n2 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
```

4행 3열 배열을 만든다.

---

### 5.5 zeros / ones

```python
np.zeros(5)
np.ones((2, 3))
```

0 또는 1로 채워진 배열을 만든다.

---

### 5.6 linspace / arange

```python
np.linspace(-1, 1, 11)
np.arange(-1, 1.2, 0.2)
```

그래프나 벡터 연산에 사용할 수치 배열을 만든다.

---

### 5.7 slicing

```python
n2[:, 0]
```

모든 행에서 첫 번째 열만 선택한다.

---

### 5.8 Boolean Indexing

```python
mask = data > 0
data[mask]
```

조건에 맞는 값만 선택한다.

---

### 5.9 reshape

```python
n9.reshape(3, 8)
n9.reshape(2, 2, -1)
```

원소 개수는 유지하면서 배열 모양만 바꾼다.

---

### 5.10 transpose

```python
n2.T
np.transpose(n10, (1, 2, 0))
```

행과 열 또는 축 순서를 바꾼다.

---

### 5.11 concatenate

```python
np.concatenate([n13, n14], axis=0)
np.concatenate([n13, n14], axis=1)
```

배열을 축 방향으로 연결한다.

---

### 5.12 Broadcasting

```python
batch_data + bias
```

2차원 배열에 1차원 bias를 자동 확장해서 더한다.

딥러닝에서 여러 데이터 샘플에 같은 편향값을 더할 때 쓰는 구조와 연결된다.

---

### 5.13 Aggregation

```python
np.sum(n1)
np.mean(n1)
np.max(n1)
np.min(n1)
np.std(n1)
np.var(n1)
```

배열의 값을 하나의 숫자나 작은 차원으로 요약한다.

---

### 5.14 Accuracy 계산

```python
matched = (yt == yp)
correct = matched.sum()
accuracy = correct / len(matched)
```

정답과 예측이 일치하는 개수를 세어 정확도를 계산한다.

---

### 5.15 Min-Max 정규화

```python
normalized = (n1 - n1.min()) / (n1.max() - n1.min())
```

값을 0과 1 사이 범위로 변환한다.

---

### 5.16 Matplotlib Line Plot

```python
plt.plot([1, 2, 3, 4])
plt.ylabel("some numbers")
plt.show()
```

값의 흐름을 선 그래프로 보여준다.

---

### 5.17 Scatter Plot

```python
plt.scatter(xs, ys)
plt.show()
```

두 변수의 관계를 점으로 보여준다.

---

### 5.18 Sigmoid Function Graph

```python
def sigmoid(x, a):
    return 1 / (1 + np.exp(-a * x))
```

시그모이드 함수 값을 계산하고 그래프로 표시한다.

---

### 5.19 Subplot

```python
ax = plt.subplot(2, 10, i + 1)
ax.imshow(img, cmap="gray_r")
```

여러 이미지를 하나의 화면에 나누어 표시한다.

---

## 6. 주요 함수 / 변수 / 약어 정리

| 이름 | 의미 | 설명 |
|---|---|---|
| `np` | NumPy | 수치 계산 라이브러리 |
| `plt` | matplotlib.pyplot | 그래프 작성 도구 |
| `ndarray` | N-dimensional array | NumPy 다차원 배열 |
| `shape` | 배열 모양 | 행, 열, 차원 크기 |
| `ndim` | number of dimensions | 배열 차원 수 |
| `dtype` | data type | 배열 원소 타입 |
| `axis` | 축 | 연산 방향 |
| `mask` | 조건 배열 | True/False로 데이터 선택 |
| `reshape` | 모양 변경 | 원소 개수 유지, 형태 변경 |
| `T` | transpose | 행과 열 변경 |
| `vstack` | vertical stack | 세로 연결 |
| `hstack` | horizontal stack | 가로 연결 |
| `concatenate` | 연결 | 배열을 지정한 축으로 연결 |
| `linspace` | linear space | 균등 간격 배열 생성 |
| `arange` | array range | step 간격 배열 생성 |
| `sum` | 합계 | 모든 값 더하기 |
| `mean` | 평균 | 중심 경향 계산 |
| `std` | standard deviation | 표준편차 |
| `var` | variance | 분산 |
| `xs` | x values | 그래프 x좌표 |
| `ys` | y values | 그래프 y좌표 |
| `xp` | x points | 함수 그래프용 x값 |
| `yp` | y points | 함수 그래프용 y값 |

---

## 7. 그래프 / 출력 결과 해석

### 7.1 Line Plot

선 그래프는 값이 어떻게 증가하거나 감소하는지 흐름을 보여준다.

### 7.2 Scatter Plot

산점도는 두 변수 간 관계를 보여준다.

확인할 수 있는 것:

- 상관관계
- 군집
- 이상치
- 분리 가능성

### 7.3 Sigmoid Graph

시그모이드 함수는 S자 모양을 가진다.

- x가 작으면 0에 가까움
- x가 크면 1에 가까움
- 기울기 계수 `a`가 크면 더 급하게 변함

### 7.4 Subplot Image

이미지 데이터도 결국 숫자 배열이다.

이미지를 `imshow()`로 출력하면 배열을 사람 눈에 보이는 그림으로 확인할 수 있다.

---

## 8. 실습에서 배운 점

- NumPy는 딥러닝 데이터 처리의 기초이다.
- Python List보다 NumPy Array가 대량 수치 연산에 적합하다.
- 반복문보다 벡터화 연산이 빠르다.
- `shape`과 `axis`를 이해해야 행렬 연산을 제대로 사용할 수 있다.
- 브로드캐스팅은 딥러닝에서 bias 연산과 연결된다.
- 집계 함수는 Loss 계산의 기초가 된다.
- Boolean Indexing은 데이터 전처리에서 중요한 필터링 도구이다.
- Matplotlib은 데이터 탐색과 모델 진단에 필요하다.
- 좋은 그래프에는 제목, 축 라벨, 범례가 필요하다.

---

## 9. 시험용 요약

- NumPy는 배열 기반 고속 수치 연산 라이브러리이다.
- NumPy 배열은 PyTorch Tensor의 기초이다.
- Python List는 범용적이지만 수치 연산에는 느리다.
- NumPy는 벡터화 연산으로 반복문 없이 배열 전체를 한 번에 계산한다.
- `shape`은 배열의 모양, `ndim`은 차원 수, `dtype`은 데이터 타입이다.
- `reshape()`는 원소 개수는 유지하면서 배열의 형태만 바꾼다.
- `.T`는 행과 열을 바꾸는 전치이다.
- `axis=0`은 세로 방향, `axis=1`은 가로 방향 집계로 이해할 수 있다.
- Boolean Indexing은 조건에 맞는 데이터만 추출한다.
- Broadcasting은 shape가 다른 배열을 자동 확장해서 연산한다.
- Aggregation은 여러 데이터를 하나의 값으로 요약한다.
- Loss는 예측값과 정답 차이를 하나의 숫자로 만드는 집계 함수로 볼 수 있다.
- Matplotlib은 데이터 분포와 모델 상태를 시각적으로 확인하는 도구이다.
- Scatter Plot은 상관관계, 군집, 이상치, 분리 가능성 확인에 사용된다.