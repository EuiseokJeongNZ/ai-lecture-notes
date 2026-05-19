# 강의_9기_AI개론_1차시_정리

## 1. 개요

이번 강의에서는 딥러닝 학습의 기본 구조와 PyTorch가 내부적으로 어떤 역할을 하는지 학습했다.

특히 다음 내용을 중심으로 진행되었다.

- 딥러닝 학습 흐름
- Loss와 Gradient 개념
- Optimizer 동작 원리
- 합성 함수 구조
- 수치 미분
- PyTorch Autograd
- 클래스(Class)와 객체지향(OOP)
- 딥러닝 모델 구조

강의에서는 수학 자체보다:

```text
"딥러닝 구조를 코드 흐름으로 이해하는 것"
```

을 매우 강조했다.

---

## 2. 오늘 배운 핵심 개념

- Forward Propagation
- Loss Function
- Backpropagation
- Gradient
- Optimizer
- Learning Rate
- Composite Function
- Numerical Differentiation
- Autograd
- Class / OOP
- __call__
- Deep Copy

---

## 3. 실습 파일 구성

| 파일 | 설명 |
|---|---|
| 01_intro_ok.ipynb | 딥러닝 구조 개념 |
| 02_python_ok.ipynb | 파이썬 함수, 미분, 클래스 |
| PDF 자료 | 강의 개념 정리 |
| SRT 스크립트 | 강사 설명 흐름 |

---

## 4. 코드 흐름 요약

```text
1. 딥러닝 학습 구조 이해
2. Loss 개념 이해
3. Optimizer 동작 이해
4. 합성 함수 구현
5. 수치 미분 구현
6. Autograd 사용
7. 클래스 실습
8. __call__ 구조 이해
9. 딥러닝 모델 구조 연결
```

---

## 5. 주요 코드 블록 설명

### Forward → Loss → Backward → Update

```python
optimizer.zero_grad()

output = model(input)

loss = fn(output, target)

loss.backward()

optimizer.step()
```

딥러닝 학습의 핵심 구조.

---

### 합성 함수

```python
y = f3(f2(f1(x)))
```

딥러닝 Layer 연결 구조와 동일.

---

### 중앙 차분


f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}


수치 미분의 핵심 공식.

---

### Autograd

```python
y.backward()
```

PyTorch 자동 미분 수행.

---

### __call__

```python
model(x)
```

객체를 함수처럼 호출 가능.

---

## 6. 주요 함수 / 변수 / 약어 정리

| 이름 | 의미 |
|---|---|
| torch | PyTorch 라이브러리 |
| tensor | PyTorch 데이터 구조 |
| nn | Neural Network |
| optim | Optimizer |
| grad | Gradient |
| lr | Learning Rate |
| epoch | 반복 학습 단위 |
| loss | 오차 |
| forward | 순전파 |
| backward | 역전파 |

---

## 7. 그래프 / 출력 결과 해석

### Loss

Loss가 작아질수록 모델이 정답에 가까워짐.

---

### Gradient

Gradient는 수정 방향과 수정 크기를 의미.

---

### tensor(12.)

Autograd가 계산한 미분값.

수치 미분 결과와 동일.

---

## 8. 실습에서 배운 점

- 딥러닝은 결국 반복적인 최적화 과정이다.
- PyTorch는 복잡한 미분 계산을 자동화한다.
- 함수 구조를 이해하면 딥러닝 Layer 구조를 이해하기 쉽다.
- 객체지향(Class) 개념이 딥러닝 모델 구조와 연결된다.
- copy()를 사용하지 않으면 원본 데이터가 오염될 수 있다.

---

## 9. 시험용 요약

```text
Forward
→ Loss
→ Backward
→ Optimizer Update
```

- Loss는 틀린 정도
- Gradient는 수정 방향
- Optimizer는 수정 수행
- 딥러닝은 합성 함수 구조
- Autograd는 자동 미분 시스템
