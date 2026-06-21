# 강의_9기_AI개론_16차시_정리  
## CNN 스크래치 구현 · SimpleCNN · SVHN · Hook · Residual Block · ResNet18 감각

## 1. 개요

이번 16차시는 사전학습 모델을 가져다 쓰는 것이 아니라 CNN을 직접 설계하고 학습하는 흐름을 정리하는 강의다.

13~15차시에서는 pretrained 모델을 많이 사용했다.  
16차시에서는 다시 기본으로 돌아가 `Conv2d`, `ReLU`, `MaxPool2d`, `Flatten`, `Linear`를 직접 쌓아 CNN을 만든다.

핵심 흐름은 다음이다.

```text
MLP가 이미지에 약한 이유
→ Local Connectivity
→ Parameter Sharing
→ Spatial Hierarchy
→ Convolution / Filter / Kernel
→ Stride / Padding
→ Feature Map
→ Activation / Pooling
→ Feature Extractor와 Classifier 분리
→ CIFAR-10 SimpleCNN 직접 구현
→ SVHN 데이터셋으로 CNN 구현
→ Sequential 방식과 class 방식 비교
→ He 초기화
→ forward hook으로 layer 출력 shape 추적
→ 하이퍼파라미터 실험
→ Residual Block과 ResNet18 기본 감각
```

이번 정리는 두 가지 실행 환경으로 나누었다.

```text
인터넷 가능 버전
→ CIFAR-10과 SVHN을 다운로드해서 강의 흐름을 따라간다.

인터넷 불가 버전
→ FakeData와 synthetic digits 데이터로 CNN 구조를 연습한다.
```

강의 원본 흐름은 거의 online 버전 기준이다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| CNN | 이미지의 공간 구조를 활용하는 합성곱 신경망 |
| MLP의 한계 | 이미지를 펼치면 위치 관계와 지역 패턴이 약해짐 |
| Local Connectivity | 가까운 픽셀끼리 먼저 연결해서 본다는 개념 |
| Parameter Sharing | 같은 필터를 이미지 전체에 재사용하는 개념 |
| Spatial Hierarchy | 낮은 층은 단순 특징, 높은 층은 복잡한 특징을 학습하는 구조 |
| Convolution | 필터와 이미지 영역을 곱하고 더하는 연산 |
| Filter / Kernel | 이미지에서 특징을 찾는 작은 행렬 |
| Stride | 필터가 이동하는 간격 |
| Padding | 가장자리에 값을 추가해 크기 변화를 조절하는 방법 |
| Feature Map | 필터를 적용해서 나온 특징 지도 |
| ReLU | 음수는 0, 양수는 그대로 통과시키는 활성화 함수 |
| Pooling | 공간 크기를 줄이고 중요한 값만 남기는 연산 |
| Feature Extractor | 이미지 특징을 뽑는 CNN 앞부분 |
| Classifier | 뽑힌 특징을 class로 분류하는 뒷부분 |
| CIFAR-10 | 32×32 RGB 10 class 이미지 데이터셋 |
| SVHN | Street View House Numbers, 숫자 이미지 데이터셋 |
| Hook | 중간 layer 출력 shape을 추적하는 기능 |
| He Initialization | ReLU 계열 모델에 적합한 초기화 |
| Residual Block | 입력을 출력에 더하는 skip connection block |
| ResNet | Residual Block을 깊게 쌓은 CNN 구조 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `Summary_Day16_online_student_note_da.ipynb` | 인터넷 가능 버전, CIFAR-10/SVHN 다운로드 기반 |
| `Summary_Day16_offline_student_note_da.ipynb` | 인터넷 불가 버전, FakeData/synthetic digits 기반 |
| `Summary_Day16_student_note_da.ipynb` | 두 버전 안내용 파일 |
| `README_AI_16차시_정리_student_note_da.md` | 16차시 개념과 코드 흐름 정리 |
| `강의_9기_AI개론_16차시_01_CNN 모델 빌드.ipynb` | CIFAR-10 SimpleCNN 기본 실습 |
| `강의_9기_AI개론_16차시_02_CNN_모델_빌드_SVHN.ipynb` | SVHN Sequential/ClassCNN 실습 |
| `강의_9기_AI개론_16차시_03_CNN_모델_빌드_실습_SVHN.ipynb` | SVHN_CNN, hook, 학습/평가/혼동행렬 실습 |
| `강의_9기_AI개론_16차시_CNN 스크래치 구현 Residual block_ResNet18.pdf` | CNN 개념, 하이퍼파라미터, 구조 설계 PDF |
| `16. CNN 스크래치 구현.srt` | 강의 스크립트 |

---

## 4. 코드 흐름 요약

### 인터넷 가능 버전

```text
1. 라이브러리 import
2. device 설정과 seed 고정
3. CNN 등장 배경과 MLP 한계 정리
4. Conv2d / MaxPool2d shape 확인
5. CIFAR-10 transform과 DataLoader 구성
6. SimpleCNN 모델 정의
7. SimpleCNN feature extractor와 classifier shape 확인
8. 학습/평가 함수 정의
9. CIFAR-10 짧은 학습 실행
10. 학습 곡선과 예측 이미지 시각화
11. SVHN transform과 DataLoader 구성
12. SVHN 샘플 이미지 확인
13. SVHN_CNN 모델 정의
14. 파라미터 수 계산
15. forward hook으로 layer 출력 shape 추적
16. SVHN 학습/평가 함수 정의
17. SVHN 짧은 학습과 best model 저장
18. 예측 결과 시각화
19. Confusion Matrix와 Classification Report 확인
20. Sequential 방식과 class 방식 비교
21. He 초기화 적용한 ClassCNN 정의
22. 하이퍼파라미터별 파라미터 수 비교
23. Residual Block 구현
24. TinyResNet 구조 확인
```

### 인터넷 불가 버전

```text
1. 라이브러리 import
2. FakeData로 CIFAR-10식 데이터 생성
3. SimpleCNN 모델 정의
4. 학습/평가 함수 정의
5. FakeData 짧은 학습 실행
6. synthetic digits 데이터 직접 생성
7. SVHN_CNN 구조를 synthetic digits에 적용
8. hook으로 layer 출력 shape 추적
9. synthetic digits 학습 실행
10. BasicResidualBlock 구조 구현
11. Confusion Matrix와 Classification Report 확인
```

---

## 5. 주요 코드 블록 설명

### 5.1 Conv2d shape 확인

```python
conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
```

RGB 3채널 이미지를 받아 32개 feature map을 만든다.

`padding=1`이 있으므로 32×32 크기가 유지된다.  
Pooling을 지나면 16×16으로 줄어든다.

---

### 5.2 SimpleCNN features

```python
self.features = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)
```

이미지에서 특징을 뽑는 부분이다.

32×32 이미지는 pooling 두 번을 지나 8×8이 된다.

---

### 5.3 SimpleCNN classifier

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

뽑힌 feature map을 1차원으로 펼친 뒤 10개 class 점수로 바꾼다.

---

### 5.4 학습 루프

```python
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

CNN 학습의 기본 순서다.

기울기를 초기화하고, 예측하고, 손실을 계산하고, 역전파하고, 파라미터를 업데이트한다.

---

### 5.5 평가 루프

```python
model.eval()

with torch.no_grad():
    outputs = model(inputs)
```

평가할 때는 Dropout/BatchNorm 동작을 평가 모드로 바꾸고 gradient 계산을 끈다.

---

### 5.6 SVHN_CNN 구조

```text
Conv1 3→32 → Pool
Conv2 32→64 → Pool
Conv3 64→128 → Pool
Flatten 128×4×4
Linear 2048→512
Dropout
Linear 512→10
```

SVHN 32×32 숫자 이미지를 10개 숫자 class로 분류한다.

---

### 5.7 파라미터 수 계산

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

학습 가능한 파라미터 수를 계산한다.

모델 크기, 계산량, 과적합 위험을 판단하는 데 도움이 된다.

---

### 5.8 forward hook

```python
handle = module.register_forward_hook(hook_fn)
```

layer가 forward될 때 중간 출력 shape을 저장한다.

복잡한 CNN에서 shape 오류를 찾는 데 매우 유용하다.

강사님은 hook을 중간 지점에 리본이나 CCTV를 달아두는 것처럼 설명했다.

---

### 5.9 Sequential 방식

```python
seq_model = nn.Sequential(seq_feature_extractor, seq_classifier)
```

순서대로 흐르는 단순한 모델에 적합하다.

코드가 짧고 빠르게 만들 수 있다.

---

### 5.10 class 방식

```python
class ClassCNN(nn.Module):
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

forward 흐름을 직접 제어할 수 있다.

hook, skip connection, 복잡한 모델 구조에 더 적합하다.

---

### 5.11 He 초기화

```python
nn.init.kaiming_normal_(m.weight)
```

ReLU 계열 활성화 함수와 잘 맞는 초기화 방식이다.

학습 초반을 더 안정적으로 만들 수 있다.

---

### 5.12 AdamW와 weight_decay

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

`weight_decay`는 L2 정규화 계열 역할을 하며 과적합을 줄이는 데 도움을 준다.

---

### 5.13 Residual Block

```python
out = F(x) + x
```

입력을 block 출력에 더하는 skip connection 구조다.

shape이 다르면 shortcut에 1×1 Conv를 넣어 channel과 크기를 맞춘다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `torch` | PyTorch 메인 라이브러리 | Tensor와 학습 |
| `nn` | neural network 모듈 | layer, loss |
| `optim` | optimizer 모듈 | Adam, AdamW |
| `F` | torch.nn.functional | `F.relu` 등 |
| `Conv2d` | 2D 합성곱 layer | `nn.Conv2d(...)` |
| `MaxPool2d` | 최대 pooling | 공간 크기 축소 |
| `Flatten` | 1차원 펼치기 | classifier 입력 |
| `Linear` | 완전연결층 | class 점수 출력 |
| `Dropout` | 일부 뉴런 비활성화 | 과적합 완화 |
| `CIFAR-10` | 10 class 컬러 이미지 데이터 | 32×32 RGB |
| `SVHN` | 거리뷰 집 번호 이미지 | 숫자 0~9 |
| `batch_size` | 한 번에 학습할 샘플 수 | 64, 128 등 |
| `epoch` | 전체 데이터 1회 학습 | 반복 단위 |
| `loss` | 손실값 | 모델 오차 |
| `accuracy` | 정확도 | 맞힌 비율 |
| `weight_decay` | 가중치 감쇠 | 과적합 완화 |
| `hook` | 중간 출력 추적 | `register_forward_hook` |
| `OrderedDict` | 순서 유지 딕셔너리 | layer shape 저장 |
| `kaiming_normal_` | He 초기화 | ReLU 계열에 적합 |
| `Residual Block` | skip connection block | `F(x)+x` |
| `shortcut` | 건너뛰는 연결 | shape 맞춤 |
| `ResNet` | Residual Network | 깊은 CNN 구조 |

---

## 7. 그래프/출력 결과 해석

### 7.1 Loss Curve

train/test loss가 함께 내려가면 학습이 정상적으로 진행되는 것이다.

train loss만 내려가고 test loss가 올라가면 과적합을 의심한다.

### 7.2 Accuracy Curve

accuracy가 올라가면 class를 더 잘 맞히는 것이다.

SVHN에서는 숫자별 이미지가 비슷할 수 있어 특정 숫자끼리 헷갈릴 수 있다.

### 7.3 예측 이미지 시각화

정답과 예측을 같이 보면 모델이 실제로 어떤 이미지를 틀리는지 확인할 수 있다.

파란색은 정답, 빨간색은 오답으로 표시한다.

### 7.4 Confusion Matrix

대각선은 맞힌 개수다.

대각선 밖은 어떤 숫자를 어떤 숫자로 착각했는지 보여준다.

### 7.5 Hook 출력 shape

Conv와 Pooling을 지날 때 Tensor shape이 어떻게 변하는지 확인한다.

shape 오류가 생기면 hook 출력으로 어느 layer에서 문제가 생겼는지 찾을 수 있다.

---

## 8. 실습에서 배운 점

- CNN은 이미지의 지역적 구조를 보존하면서 특징을 뽑는다.
- Conv2d의 출력 channel 수는 feature map 개수와 연결된다.
- Pooling은 공간 크기를 줄여 계산량을 줄인다.
- Feature Extractor와 Classifier를 분리하면 모델 구조가 이해하기 쉽다.
- `nn.Sequential`은 단순 구조에 편하다.
- class 기반 모델은 복잡한 구조에 적합하다.
- CNN에서는 shape 계산이 중요하다.
- hook을 쓰면 중간 layer의 shape을 쉽게 추적할 수 있다.
- 하이퍼파라미터에는 고정 정답이 없고 데이터에 맞게 실험해야 한다.
- Residual Block은 깊은 CNN에서 gradient 흐름을 돕는다.
- ResNet은 Residual Block을 쌓아 깊은 네트워크를 학습할 수 있게 만든 구조다.

---

## 9. 시험용 요약

```text
16강 핵심 = CNN을 직접 설계하고 layer별 shape, 파라미터, 학습 흐름을 이해하는 것
```

- MLP는 이미지를 펼쳐 공간 정보를 잃는다.
- CNN은 지역 연결, 파라미터 공유, 계층적 특징 학습을 활용한다.
- `Conv2d`의 `out_channels`는 feature map 개수다.
- `padding=1`은 3×3 Conv에서 공간 크기를 유지한다.
- `MaxPool2d(2)`는 가로세로 크기를 절반으로 줄인다.
- Feature Extractor는 이미지 특징을 뽑는 부분이다.
- Classifier는 특징을 class 점수로 바꾸는 부분이다.
- CIFAR-10은 32×32 RGB 10 class 데이터셋이다.
- SVHN은 거리뷰 집 번호 숫자 이미지 데이터셋이다.
- `nn.Sequential`은 단순 순차 모델에 편하다.
- class 방식은 forward 흐름을 직접 제어할 수 있다.
- `view(x.size(0), -1)`은 batch를 유지하고 나머지를 펼친다.
- Conv 파라미터 수는 `(k×k×in_channels + 1) × out_channels`다.
- Linear 파라미터 수는 `(in_features + 1) × out_features`다.
- hook은 중간 layer 출력 shape을 추적한다.
- He 초기화는 ReLU 계열 모델에 자주 사용된다.
- Residual Block은 `F(x) + x` 구조다.
- shape이 다르면 shortcut에 1×1 Conv를 사용한다.
- ResNet은 깊은 CNN에서 gradient 흐름을 돕는 구조다.