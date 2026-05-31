# 강의_9기_AI개론_11차시_정리  
## CNN 기반 이미지 분류 · Conv2d · Kernel/Filter · Feature Map · Pooling · Flatten · CIFAR-10 FCN vs CNN

## 1. 개요

이번 11차시는 CNN 기반 이미지 분류를 시작하는 강의다.

10차시까지는 이미지를 1차원 벡터로 펼쳐 MLP에 넣었다.  
하지만 이미지에서는 픽셀의 위치 관계, 선, 모서리, 질감, 주변 픽셀과의 연결이 중요하다.  
이미지를 처음부터 1차원으로 펼치면 이런 공간 정보가 약해진다.

CNN은 이 문제를 해결하기 위해 이미지를 2차원/3차원 구조 그대로 보고, 작은 필터를 이미지 위에서 움직이며 특징을 추출한다.

강의 핵심 흐름은 다음이다.

```text
이미지의 공간 정보 문제
→ CNN = Convolutional Neural Network
→ 특징 추출 Feature Extraction
→ 분류 Classifier
→ Kernel / Filter
→ Feature Map
→ Conv2d
→ ReLU
→ MaxPool2d
→ Flatten
→ Fully Connected Layer
→ CIFAR-10에서 FCN과 CNN 비교
```

이번 정리는 두 가지 실행 환경으로 나누었다.

```text
인터넷 가능 버전
→ MNIST와 CIFAR-10을 torchvision.datasets로 다운로드해서 강의 흐름을 따라간다.

인터넷 불가 버전
→ 합성 이미지 데이터를 직접 만들어 CNN 구조와 shape 흐름을 연습한다.
```

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| CNN | 이미지의 공간 구조를 유지하며 특징을 추출하는 신경망 |
| FCN | Fully Connected Network, 이미지를 펼쳐서 학습하는 완전 결합형 모델 |
| Feature Extraction | 이미지에서 선, 모서리, 질감 같은 특징을 뽑는 단계 |
| Classifier | 추출된 특징으로 최종 class를 판단하는 단계 |
| Kernel | 이미지 위를 움직이며 계산하는 작은 필터 |
| Filter | kernel과 거의 같은 의미로 쓰이며 feature map을 만든다 |
| Feature Map | 필터를 이미지에 적용한 결과 |
| Conv2d | 2차원 합성곱 Layer |
| Channel | 이미지의 색상 또는 특징 축 |
| RGB | 빨강, 초록, 파랑 3채널 |
| NCHW | PyTorch CNN 입력 순서, batch-channel-height-width |
| ReLU | 음수는 0, 양수는 그대로 통과시키는 활성화 함수 |
| MaxPool2d | 일정 영역에서 가장 큰 값만 남기는 pooling |
| Flatten | 다차원 feature map을 1차원 벡터로 펼치는 과정 |
| Sequential | Layer를 순서대로 묶는 PyTorch 도구 |
| CIFAR-10 | 10개 class의 32×32 RGB 이미지 데이터셋 |
| DataLoader | 데이터를 mini-batch로 공급하는 도구 |
| view | Tensor shape을 바꾸는 함수 |
| permute | Tensor 차원 순서를 바꾸는 함수 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `Summary_Day11_online_student_note_da.ipynb` | 인터넷 가능 버전, MNIST와 CIFAR-10 다운로드 기반 CNN 실습 |
| `Summary_Day11_offline_student_note_da.ipynb` | 인터넷 불가 버전, 합성 이미지 데이터 기반 CNN 구조 실습 |
| `Summary_Day11_student_note_da.ipynb` | 두 버전 사용 안내용 파일 |
| `README_AI_11차시_정리_student_note_da.md` | 11차시 개념과 코드 흐름 정리 |
| `강의_9기_AI개론_11차시_01_dl_ok.ipynb` | MNIST 기반 MLP, ReLU, GPU, Dataset, Transform, DataLoader, 은닉층 추가 |
| `강의_9기_AI개론_11차시_02_cnn_ok.ipynb` | MNIST Conv2d 실습, CIFAR-10 FCN/CNN 비교, shape 변환 |
| `11. CNN 기반 이미지 분류.srt` | 강의 스크립트 |
| `강의_9기_AI개론_11차시_(CNN기반 이미지분류).pdf` | CNN 기반 이미지 분류 개념 자료 |

---

## 4. 코드 흐름 요약

### 인터넷 가능 버전

```text
1. 라이브러리 import
2. device 설정
3. ReLU 함수 복습
4. CNN이 필요한 이유 정리
5. CNN의 2단계 구조 정리
6. MNIST 데이터 다운로드
7. MNIST 이미지 shape [1, 28, 28] 확인
8. Conv2d 입력을 [1, 1, 28, 28]로 변환
9. 대각선 3×3 필터 직접 생성
10. Conv2d를 여러 번 적용해 feature map 변화 확인
11. Conv2d / ReLU / MaxPool2d shape 변화 확인
12. Conv2d weight shape 확인
13. nn.Sequential로 특징 추출기 구성
14. nn.Flatten으로 [32,14,14]를 6272 feature로 변환
15. 공통 함수 torch_seed, eval_loss, fit, evaluate_history, show_images_labels 정의
16. CIFAR-10 FCN용 transform 정의
17. CIFAR-10 CNN용 transform 정의
18. CIFAR-10 데이터 다운로드
19. 빠른 실행용 Subset 구성
20. FCN용 데이터 [batch, 3072] 확인
21. CNN용 데이터 [batch, 3, 32, 32] 확인
22. FCN 모델 정의와 학습
23. CNN 모델 정의와 학습
24. FCN과 CNN validation accuracy 비교
25. CNN 예측 이미지 확인
26. classification_report와 confusion_matrix 확인
27. view와 permute Tensor 변환 연습
```

### 인터넷 불가 버전

```text
1. 라이브러리 import
2. ReLU 함수 복습
3. 28×28 합성 흑백 이미지 생성
4. 대각선 Conv2d 필터 적용
5. feature map 변화 시각화
6. dummy RGB Tensor로 Conv2d/ReLU/MaxPool2d shape 확인
7. Flatten으로 6272 feature 확인
8. CIFAR-10식 합성 컬러 이미지 생성
9. 합성 이미지 시각화
10. train/test 분할
11. FCN용 [N, 3072] 데이터 구성
12. CNN용 [N, 3, 32, 32] 데이터 구성
13. TensorDataset과 DataLoader 구성
14. 공통 학습/평가 함수 정의
15. FCN 모델 정의와 학습
16. CNN 모델 정의와 학습
17. FCN과 CNN test accuracy 비교
18. classification_report와 confusion_matrix 확인
19. 예측 이미지 확인
20. view와 permute 연습
```

---

## 5. 주요 코드 블록 설명

### 5.1 device 설정

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

GPU가 있으면 `cuda:0`을 사용하고, 없으면 CPU를 사용한다.

모델과 Tensor가 같은 device에 있어야 한다.

---

### 5.2 Conv2d 기본 구조

```python
nn.Conv2d(in_channels, out_channels, kernel_size)
```

- `in_channels`: 입력 channel 수다.
- `out_channels`: 만들 feature map 수다.
- `kernel_size`: 필터 크기다.

예를 들어 다음 코드는 RGB 이미지에서 32개 feature map을 만든다.

```python
nn.Conv2d(3, 32, 3)
```

---

### 5.3 Conv2d weight shape

```python
conv1.weight.shape
```

Conv2d weight shape은 다음 순서다.

```text
[out_channels, in_channels, kernel_height, kernel_width]
```

`nn.Conv2d(3, 32, 3)`이면 weight shape은 `[32, 3, 3, 3]`이다.

---

### 5.4 MNIST Conv2d 입력 shape

```python
image.view(1, 1, 28, 28)
```

MNIST 한 장은 `[1, 28, 28]`이다.

Conv2d에 넣으려면 batch 차원까지 포함해 `[1, 1, 28, 28]`이 필요하다.

---

### 5.5 대각선 필터

```python
w = torch.tensor([[0,0,1], [0,1,0], [1,0,0]]).float()
w = w.view(1, 1, 3, 3)
conv.weight.data = w
```

직접 만든 3×3 필터를 Conv2d weight로 넣는다.

이 필터는 특정 대각선 방향 특징에 반응한다.

---

### 5.6 shape 변화

```text
[100, 3, 32, 32]
→ Conv2d(3, 32, 3)
→ [100, 32, 30, 30]
→ Conv2d(32, 32, 3)
→ [100, 32, 28, 28]
→ MaxPool2d(2,2)
→ [100, 32, 14, 14]
```

padding 없이 3×3 Conv를 사용하면 H와 W가 2씩 줄어든다.

MaxPool2d(2,2)는 H와 W를 절반으로 줄인다.

---

### 5.7 nn.Sequential

```python
features = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 32, 3),
    nn.ReLU(inplace=True),
    nn.MaxPool2d((2, 2))
)
```

Layer를 순서대로 묶는다.

특징 추출기처럼 사용할 수 있다.

---

### 5.8 nn.Flatten

```python
flatten = nn.Flatten()
flat_outputs = flatten(feature_outputs)
```

batch 차원을 제외하고 나머지 차원을 모두 펼친다.

`[100, 32, 14, 14]`는 `[100, 6272]`가 된다.

---

### 5.9 CIFAR-10 FCN transform

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    transforms.Lambda(lambda x: x.view(-1))
])
```

이미지를 Tensor로 바꾸고 정규화한 뒤 1차원으로 펼친다.

완전 결합형 모델 입력용이다.

---

### 5.10 CIFAR-10 CNN transform

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
```

CNN에서는 이미지 공간 구조를 유지해야 하므로 `view(-1)`을 넣지 않는다.

---

### 5.11 FCN 모델

```python
Linear(3072 → 128)
→ ReLU
→ Linear(128 → 10)
```

CIFAR-10 이미지를 `[3072]` 벡터로 펼쳐 분류한다.

공간 정보 손실이 생길 수 있다.

---

### 5.12 CNN 모델

```python
features:
Conv2d(3 → 32)
→ ReLU
→ Conv2d(32 → 32)
→ ReLU
→ MaxPool2d

classifier:
Flatten
→ Linear(6272 → 128)
→ ReLU
→ Linear(128 → 10)
```

이미지의 공간 구조를 유지하면서 특징을 뽑고, 마지막에 분류한다.

---

### 5.13 fit 함수

```python
fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device)
```

모델 학습과 검증을 반복하는 공통 함수다.

FCN과 CNN을 같은 방식으로 비교하기 위해 사용한다.

---

### 5.14 show_images_labels 함수

```python
show_images_labels(loader, classes, net, device)
```

이미지와 정답, 예측값을 함께 보여준다.

틀린 예측은 빨간색으로 표시하도록 만들 수 있다.

---

### 5.15 view

```python
x.view(-1)
```

Tensor를 1차원으로 펼친다.

FCN 입력을 만들거나 Flatten 흐름을 확인할 때 사용한다.

---

### 5.16 permute

```python
x.permute(1, 2, 0)
```

Tensor 차원 순서를 바꾼다.

PyTorch의 CHW 이미지를 Matplotlib이 보기 쉬운 HWC 순서로 바꿀 때 자주 사용한다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `CNN` | 합성곱 신경망 | 이미지 분류에 특화 |
| `FCN` | 완전 결합형 신경망 | 이미지를 1차원으로 펼쳐 사용 |
| `DNN` | Deep Neural Network | 깊은 신경망 일반 표현 |
| `Conv2d` | 2차원 합성곱 Layer | `nn.Conv2d(3, 32, 3)` |
| `kernel` | 작은 필터 행렬 | 이미지 위를 이동 |
| `filter` | 특징을 찾는 창 | feature map 생성 |
| `feature map` | 필터 적용 결과 | 특징이 강조된 출력 |
| `channel` | 색상/특징 축 | RGB는 3채널 |
| `NCHW` | CNN 입력 차원 순서 | batch, channel, height, width |
| `ReLU` | 활성화 함수 | 음수는 0, 양수는 그대로 |
| `MaxPool2d` | 최대 풀링 | `nn.MaxPool2d((2,2))` |
| `Flatten` | 1차원 펼치기 | `nn.Flatten()` |
| `features` | 특징 추출기 | Conv/ReLU/Pool 묶음 |
| `classifier` | 분류기 | Flatten/Linear 묶음 |
| `Sequential` | 순차 실행 묶음 | `nn.Sequential(...)` |
| `CIFAR-10` | 10개 class 컬러 이미지 데이터 | 3×32×32 |
| `MNIST` | 손글씨 숫자 데이터 | 1×28×28 |
| `DataLoader` | batch 공급 도구 | `DataLoader(dataset, batch_size=...)` |
| `view` | shape 변경 | `x.view(-1)` |
| `permute` | 차원 순서 변경 | `x.permute(1,2,0)` |
| `CrossEntropyLoss` | 다중 분류 손실 | logits + long label |
| `SGD` | 확률적 경사하강법 | `optim.SGD(...)` |

---

## 7. 그래프/출력 결과 해석

### 7.1 ReLU 그래프

음수 영역은 0이 되고, 양수 영역은 그대로 올라간다.

CNN의 Conv2d 뒤에서 비선형성을 추가하는 역할을 한다.

### 7.2 MNIST Conv2d 결과

대각선 필터를 적용하면 특정 방향 특징이 강조된다.

Conv2d를 반복할수록 feature map 크기는 줄고, 특정 패턴이 더 두드러진다.

### 7.3 Conv2d shape 출력

`[100, 3, 32, 32]`가 `[100, 32, 14, 14]`로 변한다.

채널은 feature map 개수로 늘고, 공간 크기는 Conv와 Pooling으로 줄어든다.

### 7.4 Flatten 출력

`[100, 32, 14, 14]`가 `[100, 6272]`로 변한다.

이후 Linear Layer에 넣을 수 있다.

### 7.5 FCN 학습 곡선

이미지를 벡터로 펼쳐 학습하므로 공간 정보가 약해질 수 있다.

검증 정확도가 낮거나 과적합이 생길 수 있다.

### 7.6 CNN 학습 곡선

CNN은 공간 정보를 유지하며 특징을 추출한다.

이미지 분류에서는 FCN보다 더 적합한 구조다.

### 7.7 Confusion Matrix

대각선은 맞힌 class다.

대각선 밖은 어떤 class를 어떤 class로 헷갈렸는지 보여준다.

---

## 8. 실습에서 배운 점

- CNN은 이미지의 공간 구조를 유지하기 위해 등장한 구조다.
- 완전 결합형 모델은 이미지를 1차원으로 펼쳐 공간 정보를 잃을 수 있다.
- CNN은 필터/커널을 학습한다.
- 필터를 적용한 결과를 feature map이라고 한다.
- CNN은 특징 추출기와 분류기로 나누어 이해하면 쉽다.
- Conv2d는 `[N, C, H, W]` 입력을 받는다.
- RGB 이미지는 channel이 3개다.
- Conv2d의 output channel 수는 feature map 수와 연결된다.
- padding 없이 3×3 Conv를 쓰면 H와 W가 2씩 줄어든다.
- MaxPool2d는 중요한 값만 남기고 공간 크기를 줄인다.
- Flatten은 다차원 feature map을 Linear에 넣기 위해 1차원으로 펼친다.
- `nn.Sequential`은 Layer를 순서대로 묶을 때 편하다.
- FCN과 CNN은 같은 분류 문제를 풀지만 입력을 다루는 방식이 다르다.
- `view(-1)`은 Flatten과 관련된 shape 변경이다.
- `permute(1,2,0)`은 CHW 이미지를 HWC로 바꿀 때 자주 쓴다.

---

## 9. 시험용 요약

```text
CNN = 이미지의 공간 구조를 유지하면서 필터로 특징을 추출하는 신경망
```

- CNN은 Convolutional Neural Network의 약자다.
- CNN은 이미지 분류에 특화된 구조다.
- DNN/FCN은 이미지를 1차원 벡터로 펼친다.
- 이미지를 펼치면 픽셀 간 공간 정보가 약해진다.
- CNN은 작은 필터를 움직이며 지역 특징을 추출한다.
- CNN은 크게 특징 추출기와 분류기로 나뉜다.
- 특징 추출기는 Conv2d, ReLU, Pooling으로 구성된다.
- 분류기는 Flatten과 Linear로 구성된다.
- Conv2d의 기본 형태는 `nn.Conv2d(in_channels, out_channels, kernel_size)`다.
- Conv2d weight shape은 `[out_channels, in_channels, kernel_height, kernel_width]`다.
- out_channels는 feature map 개수다.
- PyTorch CNN 입력은 `[N, C, H, W]`다.
- CIFAR-10 이미지는 `[3, 32, 32]`이다.
- MNIST 이미지는 `[1, 28, 28]`이다.
- 3×3 Conv를 padding 없이 적용하면 H와 W가 2씩 줄어든다.
- MaxPool2d(2,2)는 보통 H와 W를 절반으로 줄인다.
- `[32, 14, 14]`를 Flatten하면 6272개 feature다.
- Linear Layer에 넣기 전에는 Flatten이 필요하다.
- `nn.Sequential`은 Layer를 순서대로 실행한다.
- CNN은 FCN보다 이미지의 공간 정보를 잘 활용한다.
- `view`는 Tensor shape을 바꾼다.
- `permute`는 Tensor 차원 순서를 바꾼다.