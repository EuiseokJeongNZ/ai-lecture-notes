# 강의_9기_AI개론_5차시_정리  
## 경사하강법(GD) 구현 · PyTorch 모델 구조

## 1. 개요

이번 강의는 PyTorch로 머신러닝 모델을 구현하는 전체 구조를 다룬다.

핵심은 신경망을 단순한 그림이 아니라, Tensor가 여러 Layer Function을 순서대로 통과하는 **프로그램 모델**로 이해하는 것이다.

강의에서는 머신러닝 모델을 다음처럼 설명한다.

```text
머신러닝 모델 = 레이어 함수 구조 + 학습되는 파라미터 값
```

즉, 모델은 단순한 코드 덩어리가 아니라, Tensor를 변환하는 함수들의 조합이며, 그 내부에는 학습을 통해 조정되는 weight와 bias가 들어 있다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 설명 |
|---|---|
| 신경망 개념도 | 입력층, 은닉층, 출력층으로 표현되는 설계 관점 |
| PyTorch 프로그램 모델 | Tensor가 Layer Function을 통과하는 구현 관점 |
| Layer Function | Tensor를 입력받아 Tensor를 출력하는 함수 |
| Parameter | Layer 내부에 저장되어 학습되는 값 |
| Model | Layer Function을 조합한 거대한 합성 함수 |
| Learning | 정답에 가까워지도록 Parameter를 조정하는 과정 |
| `nn.Linear` | 선형 변환 Layer |
| `nn.ReLU` | 대표 활성화 함수 |
| `nn.Sequential` | 여러 Layer를 순서대로 묶는 도구 |
| `net` | 예측 함수 |
| `criterion` | 손실 함수 |
| `optimizer` | 최적화 함수 |
| Forward | 입력이 모델을 통과해 예측값을 만드는 과정 |
| Backward | Loss에서 시작해 gradient를 계산하는 과정 |
| BatchNorm | 배치 단위로 분포를 안정화하는 기법 |
| LayerNorm | 샘플별 특징 차원을 정규화하는 기법 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_5차시_01_model_dev_ok.ipynb` | PyTorch Layer, Sequential, 모델 3종 비교, 활성화 함수 실험 |
| `05. 경사하강법(GD) 구현.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_5차시_경사하강법 구현.pdf` | 강의 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. nn.Linear, nn.ReLU Layer 생성
3. 더미 입력 Tensor 생성
4. Layer를 하나씩 통과하며 shape 확인
5. nn.Sequential로 예측 함수 구성
6. y = x² + noise 데이터 생성
7. 훈련/검증 데이터 분리
8. NumPy 배열을 Tensor로 변환
9. Net 모델 정의: Linear 1개
10. Net 학습 및 결과 확인
11. Net2 모델 정의: Linear 3개, 활성화 함수 없음
12. Net2 학습 및 결과 확인
13. Net3 모델 정의: Linear + ReLU
14. Net3 학습 및 결과 확인
15. 세 모델의 차이 비교
16. Hook, Scheduler, 초기화, Minibatch, BatchNorm 개념 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 Layer Function 정의

```python
l1 = nn.Linear(784, 128)
l2 = nn.Linear(128, 10)
relu = nn.ReLU(inplace=True)
```

28x28 이미지를 784개 숫자로 펼친 뒤, 128차원 중간 표현을 거쳐 10개 클래스 출력으로 변환한다.

---

### 5.2 더미 입력 생성

```python
inputs = torch.randn(100, 784)
```

100개의 데이터, 각 데이터는 784개의 feature를 가진다.

---

### 5.3 Layer를 순서대로 통과

```python
m1 = l1(inputs)
m2 = relu(m1)
outputs = l2(m2)
```

Tensor가 Layer Function을 순서대로 통과하며 변환된다.

```text
[100, 784] → [100, 128] → [100, 128] → [100, 10]
```

---

### 5.4 `nn.Sequential`

```python
net2 = nn.Sequential(
    l1,
    relu,
    l2
)
outputs2 = net2(inputs)
```

여러 Layer를 하나의 예측 함수처럼 묶는다.

---

### 5.5 실험 데이터 생성

```python
x = np.random.randn(100, 1) * 2.5
y = x**2 + np.random.randn(100, 1) * 0.8
```

2차 함수 형태의 데이터를 만든다.

---

### 5.6 Tensor 변환

```python
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()
```

PyTorch 모델 학습을 위해 NumPy 배열을 Tensor로 변환한다.

---

### 5.7 모델 1: 선형 회귀

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 1)

    def forward(self, x):
        x1 = self.l1(x)
        return x1
```

선형 Layer 하나만 가진 모델이다.

2차 함수 형태의 데이터를 직선으로만 예측할 수 있다.

---

### 5.8 모델 2: 활성화 함수 없는 깊은 모델

```python
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 1)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3
```

Linear Layer를 여러 개 쌓았지만 활성화 함수가 없기 때문에 여전히 선형 함수에 가깝다.

---

### 5.9 모델 3: ReLU가 있는 딥러닝 모델

```python
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3
```

Linear 사이에 ReLU를 넣어 비선형성을 추가한다.

이 모델은 2차 함수 형태의 곡선 패턴을 학습할 수 있다.

---

### 5.10 학습 루프

```python
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

PyTorch 학습의 표준 구조이다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 의미 | 설명 |
|---|---|---|
| `nn` | Neural Network | PyTorch 신경망 모듈 |
| `nn.Module` | Model Base Class | PyTorch 모델 부모 클래스 |
| `nn.Linear` | Linear Layer | 선형 변환 Layer |
| `nn.ReLU` | ReLU Activation | 비선형 활성화 함수 |
| `nn.Sequential` | Sequential Container | Layer를 순서대로 묶는 도구 |
| `net` | Prediction Function | 예측 함수, 모델 |
| `criterion` | Loss Function | 손실 함수 |
| `optimizer` | Optimizer | 파라미터 수정 도구 |
| `SGD` | Stochastic Gradient Descent | 확률적 경사하강법 |
| `MSELoss` | Mean Squared Error Loss | 평균 제곱 오차 |
| `inputs` | Input Tensor | 입력 데이터 |
| `labels` | Target Tensor | 정답 데이터 |
| `outputs` | Output Tensor | 예측값 |
| `loss` | Loss | 오차 |
| `forward` | Forward Pass | 순전파 |
| `backward` | Backward Pass | 역전파 |
| `grad` | Gradient | 기울기 |
| `lr` | Learning Rate | 학습률 |
| `epoch` | Epoch | 학습 반복 단위 |
| `BatchNorm` | Batch Normalization | 배치 정규화 |
| `LayerNorm` | Layer Normalization | 샘플별 정규화 |

---

## 7. 그래프/출력 결과 해석

### 7.1 2차 함수 데이터 산점도

데이터가 U자 형태로 분포한다.

따라서 단순 직선 모델로는 잘 맞추기 어렵다.

### 7.2 Net 결과

은닉층과 활성화 함수가 없는 선형 모델이므로 예측이 직선 형태에 가깝다.

### 7.3 Net2 결과

Linear Layer를 여러 개 쌓았지만 활성화 함수가 없으므로 여전히 선형에 가깝다.

### 7.4 Net3 결과

Linear 사이에 ReLU를 넣었기 때문에 곡선 패턴을 더 잘 따라간다.

### 7.5 학습 곡선

Loss가 감소하면 학습이 진행되고 있다는 의미이다.

---

## 8. 실습에서 배운 점

- PyTorch 모델은 Tensor가 Layer Function을 순서대로 통과하는 구조이다.
- `nn.Linear`는 입력 차원을 출력 차원으로 바꾸는 선형 함수이다.
- 앞 Layer의 출력 크기와 다음 Layer의 입력 크기는 반드시 맞아야 한다.
- `nn.Sequential`은 여러 Layer를 하나의 합성 함수처럼 묶는다.
- PyTorch 학습에는 `net`, `criterion`, `optimizer`가 필요하다.
- 학습은 예측, 손실 계산, 경사 계산, 파라미터 수정의 반복이다.
- 선형 함수만 여러 번 합성해도 결국 선형 함수이다.
- 활성화 함수는 모델에 비선형성을 부여한다.
- ReLU가 있는 모델은 곡선 형태의 데이터를 더 잘 학습할 수 있다.
- Hook은 gradient 흐름을 관찰하는 데 사용된다.
- Scheduler는 학습률을 조절한다.
- 초기화는 학습 시작점을 결정한다.
- Minibatch는 메모리 효율성과 학습 속도를 개선한다.
- BatchNorm은 학습 안정성과 gradient flow 개선에 도움을 준다.

---

## 9. 시험용 요약

```text
PyTorch 학습의 3대 요소 = net + criterion + optimizer
```

- Layer Function은 Tensor를 입력받아 Tensor를 출력하는 함수이다.
- Parameter는 Layer 내부의 weight와 bias이다.
- Model은 여러 Layer Function을 연결한 합성 함수이다.
- `nn.Linear(in, out)`은 입력 차원을 출력 차원으로 바꾼다.
- `nn.Sequential`은 여러 Layer를 순서대로 묶는다.
- 학습 4단계는 예측 계산 → 손실 계산 → 경사 계산 → 파라미터 수정이다.
- `optimizer.zero_grad()`는 gradient 초기화이다.
- `outputs = net(inputs)`는 순전파이다.
- `loss.backward()`는 역전파이다.
- `optimizer.step()`은 파라미터 업데이트이다.
- 선형 함수만 여러 번 합성해도 선형 함수이다.
- 활성화 함수는 비선형성을 추가한다.
- ReLU가 있는 모델은 곡선 패턴을 학습할 수 있다.
- Hook은 gradient 관찰 도구이다.
- BatchNorm은 학습 안정성을 높인다.
- LayerNorm은 RNN, Transformer에 자주 사용된다.