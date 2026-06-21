# 강의_9기_AI개론_13차시_정리  
## 사전학습 모델 활용 1 · Pretrained Model · Fine-Tuning · Transfer Learning · ResNet18 · VGG19-BN · MobileNetV2

## 1. 개요

이번 13차시는 사전학습 모델을 활용하는 방법을 정리하는 강의다.

지금까지는 CNN 모델을 직접 만들고 학습했다.  
이번 강의에서는 ImageNet 같은 대규모 데이터셋으로 이미 학습된 모델을 가져와서 CIFAR-10 같은 새로운 문제에 맞게 바꿔 쓴다.

핵심 흐름은 다음이다.

```text
사전학습 모델 개념
→ 파인튜닝과 전이학습 차이
→ AdaptiveAvgPool2d 구조 이해
→ CIFAR-10 데이터 준비
→ ResNet18 불러오기
→ 마지막 fc layer 교체
→ 학습과 결과 평가
→ VGG19-BN 구조 확인과 classifier[6] 교체
→ Transfer Learning 방식으로 일부 파라미터 freeze
→ MobileNetV2로 초급 이미지 분류기 만들기
→ 예측 시각화, 클래스별 성능, 모델 저장
```

이번 정리는 두 가지 실행 환경으로 나누었다.

```text
인터넷 가능 버전
→ CIFAR-10 데이터와 ImageNet 사전학습 가중치를 다운로드해서 실행한다.

인터넷 불가 버전
→ FakeData와 weights=None으로 구조 수정 방법을 연습한다.
```

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| Pretrained Model | 대규모 데이터로 미리 학습된 모델 |
| ImageNet | 사전학습에 자주 쓰이는 대규모 이미지 데이터셋 |
| Fine-Tuning | 사전학습 모델 전체 또는 많은 부분을 새 데이터로 미세 조정하는 방식 |
| Transfer Learning | 앞쪽 feature extractor는 고정하고 마지막 분류기 위주로 학습하는 방식 |
| Feature Extractor | 이미지에서 선, 질감, 형태 같은 특징을 뽑는 부분 |
| Classifier | 추출된 특징을 바탕으로 최종 class를 분류하는 부분 |
| AdaptiveAvgPool2d | 입력 크기와 무관하게 원하는 출력 크기를 만드는 pooling |
| ResNet18 | 잔차 연결을 사용하는 대표 CNN 모델 |
| VGG19-BN | VGG19에 BatchNorm이 들어간 깊은 CNN 모델 |
| MobileNetV2 | 가볍고 빠른 사전학습 CNN 모델 |
| fc | ResNet의 마지막 fully connected layer |
| classifier[6] | VGG19-BN의 마지막 Linear layer |
| classifier[1] | MobileNetV2의 마지막 Linear layer |
| requires_grad | 파라미터 학습 여부를 정하는 속성 |
| state_dict | 모델 파라미터를 담은 딕셔너리 |
| ImageNet normalization | pretrained 모델에 맞는 RGB 평균/표준편차 정규화 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `Summary_Day13_online_student_note_da.ipynb` | 인터넷 가능 버전, CIFAR-10 + ImageNet pretrained weights 기반 |
| `Summary_Day13_offline_student_note_da.ipynb` | 인터넷 불가 버전, FakeData + weights=None 구조 연습 |
| `Summary_Day13_student_note_da.ipynb` | 두 버전 안내용 파일 |
| `README_AI_13차시_정리_student_note_da.md` | 13차시 개념과 코드 흐름 정리 |
| `강의_9기_AI개론_13차시_01_transfer_learning.ipynb` | ResNet18, VGG19-BN, transfer learning 실습 |
| `강의_9기_AI개론_13차시_02_이미지분류기_초급.ipynb` | MobileNetV2 초급 이미지 분류기 실습 |
| `강의_9기_AI개론_13차시_(사전_학습_모델_활용_1).pdf` | 사전학습 모델 활용 이론 자료 |
| `13. 사전학습(Pretrained) 모델 1.srt` | 강의 스크립트 |

---

## 4. 코드 흐름 요약

### 인터넷 가능 버전

```text
1. 라이브러리 import
2. device 설정과 torch_seed 함수 정의
3. 사전학습 모델 개념 정리
4. 파인튜닝과 전이학습 차이 정리
5. AdaptiveAvgPool2d shape 실습
6. CIFAR-10 class와 ImageNet mean/std 정의
7. Resize(112), RandomHorizontalFlip, Normalize, RandomErasing transform 구성
8. CIFAR-10 데이터 다운로드
9. DataLoader 구성
10. 이미지 미리 보기 함수 작성
11. ResNet18 pretrained 모델 불러오기
12. ResNet18의 fc layer 확인
13. net.fc를 nn.Linear(fc_in_features, 10)으로 교체
14. CrossEntropyLoss와 SGD with Momentum 설정
15. ResNet18 fine-tuning 학습
16. 예측 결과 시각화
17. requires_grad=False로 transfer learning freeze 구조 구성
18. freeze된 ResNet18의 fc만 학습
19. VGG19-BN의 classifier[6] 교체 구조 확인
20. MobileNetV2의 classifier[1] 교체
21. MobileNetV2 짧은 학습
22. 클래스별 정확도 계산
23. torch.save(model.state_dict())로 모델 저장
```

### 인터넷 불가 버전

```text
1. 라이브러리 import
2. FakeData로 3×112×112 이미지 데이터 생성
3. DataLoader 구성
4. 이미지 미리 보기
5. AdaptiveAvgPool2d shape 실습
6. ResNet18(weights=None) 구조 생성
7. ResNet18 fc 교체
8. 짧은 학습 루프 실행
9. requires_grad=False로 freeze 구조 연습
10. VGG19-BN classifier[6] 교체 구조 확인
11. MobileNetV2 classifier[1] 교체
12. MobileNetV2 짧은 학습
13. 모델 state_dict 저장
```

---

## 5. 주요 코드 블록 설명

### 5.1 AdaptiveAvgPool2d

```python
p = nn.AdaptiveAvgPool2d((1, 1))
m1 = p(inputs)
m2 = m1.view(m1.shape[0], -1)
```

입력 feature map 크기와 상관없이 출력 공간 크기를 1×1로 만든다.

`[N, C, H, W]`가 `[N, C, 1, 1]`이 되고, `view`를 거쳐 `[N, C]`가 된다.

---

### 5.2 ImageNet 정규화

```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

ImageNet 사전학습 모델을 사용할 때 자주 쓰는 RGB 평균과 표준편차다.

사전학습 가중치를 제대로 활용하려면 입력 전처리를 맞추는 것이 중요하다.

---

### 5.3 ResNet18 불러오기

```python
weights = models.ResNet18_Weights.DEFAULT
net = models.resnet18(weights=weights)
```

ImageNet으로 사전학습된 ResNet18 모델을 불러온다.

예전 코드에서는 `models.resnet18(pretrained=True)`를 사용한다.

---

### 5.4 ResNet18 fc 교체

```python
fc_in_features = net.fc.in_features
net.fc = nn.Linear(fc_in_features, n_output)
```

ImageNet 1000개 class용 마지막 layer를 CIFAR-10 10개 class용 layer로 교체한다.

---

### 5.5 Fine-Tuning 학습 설정

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

모델 전체 parameter를 학습 대상으로 두고 작은 learning rate로 미세 조정한다.

---

### 5.6 Transfer Learning freeze

```python
for param in net.parameters():
    param.requires_grad = False

net.fc = nn.Linear(net.fc.in_features, n_output)
```

기존 feature extractor는 고정하고, 새로 교체한 마지막 layer만 학습한다.

---

### 5.7 VGG19-BN classifier 교체

```python
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, n_output)
```

VGG19-BN은 마지막 Linear layer가 `classifier[6]`에 있다.

ResNet처럼 `fc`가 아니므로 구조를 먼저 확인해야 한다.

---

### 5.8 VGG19-BN 구조 조정

```python
net.features = net.features[:-1]
net.avgpool = nn.Identity()
```

112×112 실습 입력에 맞춰 마지막 pooling을 제거하고 `avgpool`을 아무 일도 하지 않는 layer로 바꾸는 구조다.

---

### 5.9 MobileNetV2 classifier 교체

```python
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 10)
```

MobileNetV2의 마지막 Linear layer는 `classifier[1]`에 있다.

---

### 5.10 클래스별 정확도

```python
class_correct[label] += correct[i].item()
class_total[label] += 1
```

각 class마다 맞힌 개수와 전체 개수를 따로 누적해 class별 성능을 계산한다.

---

### 5.11 모델 저장

```python
torch.save(model.state_dict(), model_path)
```

모델의 학습된 parameter를 저장한다.

나중에 같은 모델 구조를 만든 뒤 `load_state_dict()`로 불러온다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `pretrained` | 사전학습된 | 이미 학습된 가중치 |
| `weights` | 사전학습 가중치 객체 | `models.ResNet18_Weights.DEFAULT` |
| `ImageNet` | 대규모 이미지 데이터셋 | 사전학습 기준 데이터 |
| `fine-tuning` | 파인튜닝 | 전체 또는 많은 파라미터 미세 조정 |
| `transfer learning` | 전이학습 | 일부 고정 후 마지막 layer 중심 학습 |
| `requires_grad` | gradient 계산 여부 | `False`면 freeze |
| `AdaptiveAvgPool2d` | 출력 크기 고정 pooling | `nn.AdaptiveAvgPool2d((1,1))` |
| `ResNet18` | 잔차 연결 기반 CNN | `models.resnet18(...)` |
| `VGG19-BN` | VGG19 + BatchNorm | `models.vgg19_bn(...)` |
| `MobileNetV2` | 경량 CNN 모델 | `models.mobilenet_v2(...)` |
| `fc` | fully connected layer | ResNet 마지막 layer |
| `classifier[6]` | VGG 마지막 Linear | VGG19-BN 마지막 분류기 |
| `classifier[1]` | MobileNetV2 마지막 Linear | MobileNetV2 마지막 분류기 |
| `in_features` | Linear 입력 feature 수 | layer 교체에 사용 |
| `out_features` | Linear 출력 class 수 | CIFAR-10은 10 |
| `CrossEntropyLoss` | 다중 분류 손실 | logits + class index |
| `SGD` | 최적화 함수 | `optim.SGD(...)` |
| `momentum` | 이전 업데이트 방향 반영 | `momentum=0.9` |
| `state_dict` | 모델 파라미터 딕셔너리 | 저장/불러오기 |
| `FakeData` | 가짜 이미지 데이터셋 | 인터넷 불가 구조 연습 |

---

## 7. 그래프/출력 결과 해석

### 7.1 학습 손실 그래프

train loss와 test loss가 함께 내려가면 학습이 안정적으로 진행되는 것이다.

### 7.2 정확도 그래프

train accuracy와 test accuracy가 함께 올라가면 일반화가 어느 정도 잘 되는 것이다.

### 7.3 예측 이미지 시각화

정답과 예측 class를 같이 보면 모델이 어떤 이미지를 잘 맞히고 어떤 이미지를 헷갈리는지 볼 수 있다.

### 7.4 클래스별 정확도

전체 정확도만 보면 놓칠 수 있는 class별 약점을 확인할 수 있다.

예를 들어 자동차는 잘 맞히지만 고양이는 못 맞힐 수 있다.

---

## 8. 실습에서 배운 점

- 사전학습 모델은 큰 데이터로 이미 학습된 모델이다.
- 처음부터 학습하는 것보다 빠르고 높은 성능을 기대할 수 있다.
- 파인튜닝은 전체 또는 많은 파라미터를 새 데이터에 맞게 조정한다.
- 전이학습은 feature extractor를 고정하고 마지막 layer 중심으로 학습한다.
- 사전학습 모델은 마지막 layer가 ImageNet 1000개 class 기준이다.
- CIFAR-10에 쓰려면 마지막 layer를 10개 출력으로 교체해야 한다.
- ResNet18의 마지막 layer는 `fc`다.
- VGG19-BN의 마지막 layer는 `classifier[6]`이다.
- MobileNetV2의 마지막 layer는 `classifier[1]`이다.
- `AdaptiveAvgPool2d`는 입력 크기에 상관없이 일정한 출력 크기를 만든다.
- ImageNet pretrained weights를 쓸 때는 ImageNet mean/std 정규화가 중요하다.
- 모델 저장은 `state_dict` 방식으로 한다.

---

## 9. 시험용 요약

```text
13강 핵심 = 이미 학습된 CNN 모델을 가져와 마지막 layer를 내 문제에 맞게 바꾸는 것
```

- 사전학습 모델은 ImageNet 같은 큰 데이터로 미리 학습된 모델이다.
- 사전학습 모델은 이미지의 기본 특징을 이미 알고 있다.
- Fine-Tuning은 모델 전체 또는 많은 부분을 새 데이터로 다시 학습한다.
- Transfer Learning은 앞부분을 고정하고 마지막 분류기 위주로 학습한다.
- `requires_grad=False`는 해당 파라미터를 학습하지 않겠다는 뜻이다.
- `AdaptiveAvgPool2d((1,1))`은 `[N,C,H,W]`를 `[N,C,1,1]`로 만든다.
- CIFAR-10은 10개 class다.
- ImageNet pretrained 모델의 기존 출력은 보통 1000개 class다.
- ResNet18의 마지막 layer는 `net.fc`다.
- ResNet18은 `net.fc = nn.Linear(net.fc.in_features, 10)`으로 바꾼다.
- VGG19-BN의 마지막 layer는 `net.classifier[6]`이다.
- MobileNetV2의 마지막 layer는 `model.classifier[1]`이다.
- 사전학습 모델에는 ImageNet mean/std 정규화를 맞추는 것이 좋다.
- `CrossEntropyLoss`는 다중 분류 손실함수다.
- 파인튜닝에는 작은 learning rate와 SGD with Momentum을 많이 사용한다.
- 모델 저장은 `torch.save(model.state_dict(), path)`로 한다.