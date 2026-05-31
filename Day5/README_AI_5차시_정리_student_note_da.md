# 강의_9기_AI개론_5차시_정리  
## 경사하강법 구현 · 예측 함수 · Layer 함수 · Gradient 안정성 · BatchNorm

## 1. 개요

이번 5차시는 PyTorch로 모델을 만든다는 것이 실제 코드에서는 어떤 구조인지 정리하는 강의다.

4차시에서 `W`, `B`를 직접 만들고 경사하강법을 구현했다면, 5차시에서는 그 구조를 `nn.Linear`, `nn.ReLU`, `nn.Sequential`, `optimizer`, `DataLoader`, `BatchNorm`으로 확장한다.

핵심 흐름은 다음이다.

```text
예측 함수 = Layer 함수들의 합성 함수
학습 = Layer 내부 Parameter를 손실이 줄어드는 방향으로 수정하는 과정
```

이번 강의는 신경망을 그림으로만 보는 것이 아니라, Tensor가 여러 Layer 함수를 통과하면서 어떤 shape으로 변하는지 보는 것이 중요하다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| 예측 함수 | 입력 Tensor를 받아 예측 Tensor를 출력하는 모델 |
| Layer 함수 | Tensor를 입력받아 Tensor를 출력하는 함수 |
| Parameter | Layer 내부에서 학습되는 weight와 bias |
| Model | Layer 함수들을 조합한 큰 합성 함수 |
| Learning | Parameter를 정답에 가까워지도록 수정하는 과정 |
| nn.Linear | 입력 feature 수를 출력 feature 수로 바꾸는 선형 Layer |
| nn.ReLU | 음수는 0, 양수는 그대로 통과시키는 활성화 함수 |
| nn.Sequential | 여러 Layer를 순서대로 묶는 도구 |
| criterion | 손실 함수 |
| optimizer | 최적화 함수 |
| argmax | 가장 큰 값의 위치를 찾는 함수 |
| Hook | gradient 흐름을 관찰하는 기능 |
| Scheduler | learning rate를 자동으로 조절하는 도구 |
| Batch Size | 한 번에 학습할 데이터 개수 |
| BatchNorm | batch 기준으로 분포를 안정화하는 정규화 |
| LayerNorm | 샘플 내부 feature 기준 정규화 |
| He 초기화 | ReLU 계열에 적합한 초기화 |
| Xavier 초기화 | Tanh/Sigmoid 계열에 적합한 초기화 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `강의_9기_AI개론_5차시_01_model_dev_ok.ipynb` | Layer 함수, `nn.Sequential`, 세 가지 모델 비교, 활성화 함수 필요성 |
| `강의_9기_AI개론_5차시_02_Grad 분포와 Scheduler.ipynb` | Gradient Hook, `argmax`, `item`, Scheduler, He 초기화 |
| `강의_9기_AI개론_5차시_03_BatchNorm과 Gradient 안정성.ipynb` | Batch size, BatchNorm, Gradient 안정성, train/eval 모드 |
| `05. 경사하강법(GD) 구현.srt` | 강사님 설명 스크립트 |
| `강의_9기_AI개론_5차시_경사하강법 구현.pdf` | 강의 개념 자료 |

---

## 4. 코드 흐름 요약

```text
1. 라이브러리 import
2. Layer 함수, Parameter, Model, Learning 용어 정리
3. nn.Linear(784, 128), nn.Linear(128, 10) 생성
4. 더미 입력 Tensor [100, 784] 생성
5. Linear → ReLU → Linear 순전파 흐름 확인
6. nn.Sequential로 예측 함수를 하나로 묶기
7. net, criterion, optimizer의 역할 정리
8. 2차 함수 데이터 생성
9. 선형 모델 Net 정의
10. 활성화 함수 없는 깊은 모델 Net2 정의
11. ReLU가 있는 딥러닝 모델 Net3 정의
12. 세 모델을 같은 학습 루프로 학습
13. loss 곡선 비교
14. 테스트 데이터에서 예측 곡선 비교
15. argmax, item, max 차이 정리
16. Gradient Hook으로 gradient 평균/표준편차 확인
17. make_classification으로 분류 데이터 생성
18. TensorDataset과 DataLoader로 mini-batch 구성
19. He 초기화를 적용한 MLP 모델 생성
20. OneCycleLR Scheduler 적용
21. epoch별 gradient, accuracy, learning rate 기록
22. Batch size별 학습 loss 비교
23. BatchNorm 없는 모델과 있는 모델 비교
24. Hook으로 BatchNorm 전후 gradient 안정성 비교
25. BatchNorm의 train/eval 모드 차이 확인
26. Xavier와 He 초기화 비교
27. BatchNorm과 LayerNorm 차이 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 nn.Linear

```python
l1 = nn.Linear(784, 128)
l2 = nn.Linear(128, 10)
```

첫 번째 Layer는 784개 입력을 128개 출력으로 바꾼다.

두 번째 Layer는 128개 입력을 10개 출력으로 바꾼다.

앞 Layer의 출력 수와 다음 Layer의 입력 수가 맞아야 한다.

---

### 5.2 nn.ReLU

```python
relu = nn.ReLU()
```

음수는 0으로 바꾸고 양수는 그대로 통과시킨다.

신경망에 비선형성을 넣는 대표적인 활성화 함수다.

---

### 5.3 Forward 흐름

```python
m1 = l1(inputs)
m2 = relu(m1)
outputs = l2(m2)
```

입력 Tensor가 첫 번째 Linear, ReLU, 두 번째 Linear를 순서대로 통과한다.

shape은 `[100, 784] → [100, 128] → [100, 128] → [100, 10]`으로 변한다.

---

### 5.4 nn.Sequential

```python
net = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

여러 Layer를 순서대로 묶어 하나의 예측 함수처럼 사용한다.

---

### 5.5 PyTorch 학습의 3대 요소

```python
net = ...
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
```

- `net`: 예측 함수다.
- `criterion`: 손실 함수다.
- `optimizer`: 파라미터를 수정하는 함수다.

---

### 5.6 학습 4단계

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

PyTorch 학습의 가장 기본 구조다.

`zero_grad`로 이전 gradient를 지우고, 예측과 손실을 계산한 뒤, 역전파와 업데이트를 수행한다.

---

### 5.7 세 가지 모델 비교

```python
Net   = Linear(1, 1)
Net2  = Linear → Linear → Linear
Net3  = Linear → ReLU → Linear → ReLU → Linear
```

선형 모델과 활성화 함수 없는 깊은 모델은 결국 직선 형태만 잘 만든다.

ReLU가 있는 모델은 2차 함수 같은 비선형 패턴을 더 잘 따라갈 수 있다.

---

### 5.8 argmax와 item

```python
scores.argmax()
scores.argmax().item()
out.argmax(1)
```

- `argmax()`는 가장 큰 값의 위치를 반환한다.
- `item()`은 원소 하나짜리 Tensor를 Python 숫자로 꺼낸다.
- `out.argmax(1)`은 각 행에서 가장 큰 class index를 구한다.

---

### 5.9 Gradient Hook

```python
param.register_hook(hook_function)
```

역전파 중 특정 파라미터의 gradient를 관찰할 수 있다.

gradient 평균이나 표준편차를 확인해서 기울기 소실과 폭발을 점검할 수 있다.

---

### 5.10 DataLoader

```python
dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

입력과 정답 Tensor를 묶은 뒤 mini-batch 단위로 꺼낸다.

mini-batch 학습은 메모리와 학습 효율을 위해 중요하다.

---

### 5.11 He 초기화

```python
nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
```

ReLU 계열 활성화 함수와 잘 맞는 초기화 방법이다.

초기화가 잘못되면 gradient가 사라지거나 폭발할 수 있다.

---

### 5.12 OneCycleLR Scheduler

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    steps_per_epoch=len(train_loader),
    epochs=5
)
```

학습률을 자동으로 조절한다.

초반에는 빠르게 움직이고 후반에는 더 정밀하게 조정하는 전략으로 이해하면 된다.

---

### 5.13 Batch Size 비교

```python
batch_sizes = [8, 32, 128, 512]
```

작은 batch는 gradient 노이즈가 많지만 일반화에 도움이 될 수 있다.

큰 batch는 안정적이지만 메모리 사용량이 크고 일반화가 떨어질 수 있다.

---

### 5.14 BatchNorm

```python
nn.BatchNorm1d(64)
```

각 Layer의 입력 분포를 안정화한다.

보통 `Linear → BatchNorm → ReLU` 순서로 사용한다.

---

### 5.15 train/eval 모드

```python
model.train()
model.eval()
```

BatchNorm은 학습 모드와 평가 모드에서 사용하는 통계가 다르다.

학습할 때는 현재 batch 통계를 사용하고, 평가할 때는 누적된 running 통계를 사용한다.

---

### 5.16 Xavier와 He 초기화

```python
nn.init.xavier_normal_(layer.weight)
nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
```

Xavier는 Tanh/Sigmoid 계열에, He는 ReLU 계열에 자주 사용한다.

---

### 5.17 BatchNorm과 LayerNorm

```python
nn.BatchNorm1d(10)
nn.LayerNorm(10)
```

BatchNorm은 batch 기준 통계를 사용한다.

LayerNorm은 각 샘플 내부 feature 기준으로 정규화한다.

Transformer 계열에서는 LayerNorm을 자주 본다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `net` | 예측 함수 | 입력을 받아 예측값을 출력한다 |
| `criterion` | 손실 함수 | 예측값과 정답을 비교한다 |
| `optimizer` | 최적화 함수 | 파라미터를 업데이트한다 |
| `Layer` | Tensor 변환 함수 | `nn.Linear`, `nn.ReLU` |
| `Parameter` | 학습되는 값 | weight, bias |
| `nn.Linear` | 선형 Layer | `nn.Linear(in, out)` |
| `nn.ReLU` | 활성화 함수 | 음수는 0, 양수는 그대로 |
| `nn.Sequential` | Layer 묶음 | 순서대로 실행되는 모델 |
| `forward` | 순전파 | 입력에서 출력까지 계산 |
| `backward` | 역전파 | 손실에서 gradient 계산 |
| `loss` | 손실 | 예측이 얼마나 틀렸는지 |
| `grad` | gradient | 파라미터 수정 방향 |
| `argmax` | 최댓값 위치 | `out.argmax(1)` |
| `item` | Python 숫자 추출 | `loss.item()` |
| `DataLoader` | mini-batch 생성 | `DataLoader(dataset, batch_size=...)` |
| `TensorDataset` | Tensor 묶음 | 입력과 정답을 묶는다 |
| `Batch Size` | 한 번에 학습할 데이터 수 | 32, 64, 128 등 |
| `Scheduler` | 학습률 조절기 | `OneCycleLR`, `StepLR` |
| `Hook` | gradient 관찰 도구 | `register_hook()` |
| `BatchNorm` | batch 기준 정규화 | `nn.BatchNorm1d()` |
| `LayerNorm` | 샘플 feature 기준 정규화 | `nn.LayerNorm()` |
| `He 초기화` | ReLU용 초기화 | `kaiming_normal_()` |
| `Xavier 초기화` | Tanh/Sigmoid용 초기화 | `xavier_normal_()` |

---

## 7. 그래프/출력 결과 해석

### 7.1 2차 함수 산점도

점들이 직선이 아니라 U자 형태에 가깝다.

이 데이터는 단순 선형 모델로는 잘 맞추기 어렵다.

### 7.2 세 모델의 학습 곡선

Loss가 낮을수록 훈련 데이터에 더 잘 맞는다.

활성화 함수가 있는 모델은 비선형 패턴을 더 잘 학습할 수 있다.

### 7.3 세 모델의 예측 곡선

선형 모델과 활성화 함수 없는 깊은 모델은 직선에 가깝다.

ReLU가 있는 모델은 곡선 형태를 어느 정도 따라간다.

### 7.4 Gradient Magnitude 그래프

gradient가 너무 작으면 기울기 소실을 의심한다.

gradient가 너무 크면 기울기 폭발이나 학습 불안정을 의심한다.

### 7.5 Learning Rate 그래프

Scheduler가 learning rate를 어떻게 조절하는지 보여준다.

### 7.6 Batch Size 그래프

batch size에 따라 loss 감소가 다르게 나타날 수 있다.

작은 batch는 흔들림이 있고, 큰 batch는 상대적으로 안정적일 수 있다.

### 7.7 BatchNorm 그래프

BatchNorm이 있는 모델이 더 안정적으로 loss를 줄이면 정규화 효과가 있다고 볼 수 있다.

---

## 8. 실습에서 배운 점

- PyTorch 모델은 Layer 함수들의 합성 함수다.
- Layer 안의 parameter가 학습으로 바뀐다.
- 앞 Layer의 출력 크기와 다음 Layer의 입력 크기가 맞아야 한다.
- `nn.Sequential`은 간단한 모델을 만들 때 편하다.
- PyTorch 학습은 `net`, `criterion`, `optimizer` 세 가지가 기본이다.
- 학습 루프는 `zero_grad → forward → loss → backward → step` 순서다.
- 활성화 함수가 없으면 깊은 모델도 선형 모델과 비슷해진다.
- ReLU가 들어가면 비선형 패턴을 학습할 수 있다.
- `argmax(1)`은 batch별 class 예측에 자주 쓰인다.
- Hook은 gradient 흐름을 관찰할 때 사용한다.
- Scheduler는 learning rate를 자동으로 조절한다.
- He 초기화는 ReLU 계열에 적합하다.
- Batch size는 학습 안정성, 속도, 일반화에 영향을 준다.
- BatchNorm은 각 층의 입력 분포와 gradient 흐름을 안정화한다.
- BatchNorm이 있는 모델은 `train()`과 `eval()` 구분이 중요하다.
- LayerNorm은 RNN이나 Transformer에서 자주 사용된다.

---

## 9. 시험용 요약

```text
PyTorch 모델 = Layer 함수들을 순서대로 연결한 예측 함수
```

- Layer 함수는 Tensor를 입력받아 Tensor를 출력한다.
- Parameter는 Layer 내부에서 학습되는 weight와 bias다.
- Model은 Layer 함수들을 조합한 큰 합성 함수다.
- `nn.Linear(in, out)`에서 앞 Layer의 out과 다음 Layer의 in이 맞아야 한다.
- `nn.Sequential()`은 Layer들을 순서대로 묶는다.
- PyTorch 학습의 3대 요소는 `net`, `criterion`, `optimizer`다.
- 학습 4단계는 예측, 손실, 경사, 업데이트다.
- `optimizer.zero_grad()`는 gradient 초기화다.
- `loss.backward()`는 역전파다.
- `optimizer.step()`은 파라미터 업데이트다.
- 활성화 함수가 없으면 Linear를 여러 번 쌓아도 결국 선형이다.
- ReLU가 있으면 비선형 패턴을 학습할 수 있다.
- `argmax(1)`은 각 행에서 가장 큰 class index를 반환한다.
- `item()`은 원소 하나짜리 Tensor를 Python 숫자로 꺼낸다.
- Hook은 gradient 흐름을 관찰하는 도구다.
- Scheduler는 learning rate를 조절한다.
- He 초기화는 ReLU 계열에 적합하다.
- Xavier 초기화는 Tanh/Sigmoid 계열에 적합하다.
- Mini-batch는 데이터를 작은 묶음으로 나누어 학습하는 방식이다.
- BatchNorm은 각 층의 입력 분포를 안정화한다.
- BatchNorm은 gradient 흐름과 학습 안정성에 도움을 줄 수 있다.
- BatchNorm이 있는 모델은 `model.train()`과 `model.eval()` 구분이 중요하다.
- LayerNorm은 RNN, Transformer 계열에서 자주 쓰인다.