# 강의_9기_AI개론_15차시_정리  
## 사용자 데이터 분류 · Custom Dataset · ImageFolder · Transfer Learning · ResNet18 Fine-tuning

## 1. 개요

이번 15차시는 사용자가 직접 준비한 이미지 데이터로 분류 모델을 만드는 방법을 정리하는 강의다.

13~14차시에서는 CIFAR-10 같은 정해진 데이터셋과 사전학습 모델을 사용했다.  
15차시에서는 실제 프로젝트처럼 폴더에 직접 들어 있는 이미지 파일을 PyTorch Dataset으로 바꾸고, 사전학습 모델을 연결하는 흐름을 다룬다.

핵심 흐름은 다음이다.

```text
문제 정의
→ 사용자 이미지 데이터 준비
→ ImageFolder가 요구하는 폴더 구조 이해
→ train / val 폴더와 class 폴더 구성
→ transforms로 전처리와 데이터 증강 정의
→ ImageFolder로 Dataset 생성
→ DataLoader로 mini-batch 공급
→ VGG19-BN fine-tuning
→ VGG19-BN transfer learning
→ dog vs wolf 사용자 데이터 분류
→ ResNet18 fine-tuning
→ learning rate scheduler
→ Early Stopping
→ 학습 곡선, 혼동 행렬, 샘플 예측 시각화
```

이번 정리는 두 가지 실행 환경으로 나누었다.

```text
인터넷 가능 버전
→ hymenoptera_data, dog_wolf, CIFAR-10, pretrained weights를 다운로드해서 강의 흐름을 따라간다.

인터넷 불가 버전
→ 직접 이미지 폴더를 생성하고 작은 CNN과 weights=None 모델로 구조를 연습한다.
```

강의 원본 흐름은 거의 online 버전 기준이다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| Custom Dataset | 사용자가 직접 준비한 데이터셋 |
| ImageFolder | 폴더 구조를 기반으로 이미지 Dataset을 자동 생성하는 기능 |
| Directory Structure | train/val/test와 class 폴더를 나누는 구조 |
| Transform | 이미지 전처리와 증강을 묶는 과정 |
| Data Augmentation | 학습 이미지에 랜덤 변형을 주어 일반화 성능을 높이는 방법 |
| DataLoader | Dataset을 mini-batch 단위로 공급하는 도구 |
| Fine-tuning | 사전학습 모델 전체를 새 데이터에 맞게 미세 조정하는 방법 |
| Transfer Learning | feature extractor는 동결하고 마지막 분류기만 학습하는 방법 |
| VGG19-BN | BatchNorm이 들어간 VGG19 사전학습 모델 |
| classifier[6] | VGG19-BN의 마지막 Linear layer |
| ResNet18 | 잔차 연결 기반 사전학습 CNN 모델 |
| fc | ResNet18의 마지막 fully connected layer |
| CosineAnnealingLR | learning rate를 cosine 곡선처럼 감소시키는 scheduler |
| Early Stopping | 검증 손실 개선이 없으면 학습을 중단하는 기법 |
| Confusion Matrix | class별 예측 오류를 보여주는 표 |
| Classification Report | precision, recall, f1-score를 보여주는 평가표 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `Summary_Day15_online_student_note_da.ipynb` | 인터넷 가능 버전, 강의 원본 흐름 중심 |
| `Summary_Day15_offline_student_note_da.ipynb` | 인터넷 불가 버전, 직접 생성한 ImageFolder 구조 중심 |
| `Summary_Day15_student_note_da.ipynb` | 두 버전 안내용 파일 |
| `README_AI_15차시_정리_student_note_da.md` | 15차시 개념과 코드 흐름 정리 |
| `강의_9기_AI개론_15차시_01_custom_dl.ipynb` | ImageFolder, hymenoptera, dog/wolf, VGG19-BN 전이학습 실습 |
| `강의_9기_AI개론_15차시_02_ResNet18_파인튜닝.ipynb` | ResNet18, scheduler, early stopping, confusion matrix 실습 |
| `강의_9기_AI개론_15차시_사용자 데이터 분류 Custom Dataset 및 Transfer Learning.pdf` | 사용자 데이터 분류 이론 자료 |
| `15. 사용자 데이터 분류.srt` | 강의 스크립트 |

---

## 4. 코드 흐름 요약

### 인터넷 가능 버전

```text
1. 라이브러리 import
2. device 설정과 seed 고정
3. ImageFolder 폴더 구조 이해
4. hymenoptera_data.zip 다운로드와 압축 해제
5. train/test transform 정의
6. ImageFolder로 train_data, train_data_no_aug, val_data 생성
7. DataLoader 구성
8. 이미지 미리보기 함수 정의
9. 공통 학습/평가 함수 정의
10. VGG19-BN fine-tuning 구조 정의
11. VGG19-BN transfer learning 구조 정의
12. fine-tuning과 transfer learning 차이 정리
13. dog_wolf.zip 다운로드와 압축 해제
14. dog/wolf ImageFolder와 DataLoader 구성
15. dog/wolf transfer learning 구조 정의
16. ResNet18 fine-tuning 전체 흐름 정리
17. CIFAR-10 transform과 DataLoader 함수 정의
18. ResNet18 fc 교체 함수 정의
19. CrossEntropyLoss, SGD with Momentum, CosineAnnealingLR 설정
20. EarlyStopping 클래스 정의
21. ResNet18 train/validate 함수 정의
22. 학습 곡선 시각화 함수 정의
23. Confusion Matrix와 Classification Report 함수 정의
24. 샘플 예측 시각화 함수 정의
```

### 인터넷 불가 버전

```text
1. 라이브러리 import
2. offline_animals/train/val/class 폴더 생성
3. ants/bees 패턴 이미지 직접 생성
4. train/test transform 정의
5. ImageFolder로 Dataset 생성
6. DataLoader 구성
7. 작은 CNN 모델 정의
8. 학습/평가 함수 정의
9. 작은 CNN 학습 실행
10. loss/accuracy 그래프 출력
11. Confusion Matrix와 Classification Report 출력
12. ResNet18(weights=None) fc 교체 구조 연습
13. EarlyStopping 구조 연습
```

---

## 5. 주요 코드 블록 설명

### 5.1 ImageFolder 폴더 구조

```text
root/
  train/
    ants/
    bees/
  val/
    ants/
    bees/
```

`ImageFolder`는 `ants`, `bees` 같은 폴더 이름을 class label로 사용한다.

---

### 5.2 ImageFolder 생성

```python
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
test_data = datasets.ImageFolder(test_dir, transform=test_transform)
```

폴더 안의 이미지를 자동으로 읽고 label을 붙여 Dataset으로 만든다.

---

### 5.3 class_to_idx

```python
train_data.class_to_idx
```

class 이름이 어떤 숫자 label로 바뀌었는지 보여준다.

예를 들어 `{"ants": 0, "bees": 1}` 형태다.

---

### 5.4 학습용 transform

```python
transforms.RandomResizedCrop(224)
transforms.RandomHorizontalFlip()
transforms.RandomErasing(...)
```

학습 데이터에 랜덤 변형을 주어 과적합을 줄인다.

---

### 5.5 검증용 transform

```python
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize(...)
```

검증 데이터에는 랜덤성을 넣지 않는다.

평가 기준이 매번 바뀌면 안 되기 때문이다.

---

### 5.6 DataLoader

```python
DataLoader(train_data, batch_size=10, shuffle=True)
```

Dataset을 batch 단위로 공급한다.

학습 데이터는 `shuffle=True`로 섞어 순서 암기를 막는다.

---

### 5.7 VGG19-BN classifier 교체

```python
in_features = net.classifier[6].in_features
net.classifier[6] = nn.Linear(in_features, 2)
```

VGG19-BN의 마지막 layer를 ants/bees 2 class 분류에 맞게 교체한다.

---

### 5.8 Fine-tuning Optimizer

```python
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

모델 전체 파라미터를 optimizer에 넘긴다.

전체 모델을 미세 조정하는 방식이다.

---

### 5.9 Transfer Learning Freeze

```python
for param in net.parameters():
    param.requires_grad = False
```

기존 사전학습 가중치를 업데이트하지 않도록 동결한다.

---

### 5.10 Transfer Learning Optimizer

```python
optimizer = optim.SGD(net.classifier[6].parameters(), lr=0.001, momentum=0.9)
```

마지막 classifier만 optimizer에 넘긴다.

데이터가 적을 때 효율적인 방식이다.

---

### 5.11 dog vs wolf 사용자 데이터

```python
classes = ["dog", "wolf"]
batch_size = 5
```

데이터가 매우 적고 class가 비슷해 transfer learning이 적합한 예제다.

---

### 5.12 ResNet18 fc 교체

```python
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
```

ImageNet 1000 class용 ResNet18을 CIFAR-10 10 class용으로 바꾼다.

---

### 5.13 CosineAnnealingLR

```python
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
```

learning rate를 cosine 곡선처럼 부드럽게 감소시킨다.

초반에는 비교적 크게 학습하고, 후반에는 아주 작게 조정한다.

---

### 5.14 EarlyStopping

```python
early_stopping = EarlyStopping(patience=5, delta=0.001, path="resnet18_best.pth")
early_stopping(val_loss, model)
```

검증 손실이 개선되지 않으면 counter를 올리고, patience를 넘으면 학습을 중단한다.

가장 좋은 모델은 지정한 path에 저장한다.

---

### 5.15 Confusion Matrix

```python
cm = confusion_matrix(y_true, y_pred)
```

정답 class와 예측 class의 관계를 표로 보여준다.

어떤 class끼리 헷갈렸는지 확인할 수 있다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `Custom Dataset` | 사용자가 직접 준비한 데이터 | 이미지 폴더, CSV 등 |
| `ImageFolder` | 폴더 구조 기반 이미지 Dataset | `datasets.ImageFolder(root)` |
| `root` | 데이터 상위 폴더 | class 폴더가 들어 있는 위치 |
| `train_dir` | 학습 데이터 폴더 | `root/train` |
| `val_dir` | 검증 데이터 폴더 | `root/val` |
| `class_to_idx` | class 이름과 label 매핑 | 자동 생성 |
| `transform` | 이미지 전처리 | Resize, Crop, ToTensor 등 |
| `RandomResizedCrop` | 랜덤 crop 후 resize | 학습 증강 |
| `CenterCrop` | 중앙 crop | 검증 전처리 |
| `RandomErasing` | 일부 영역 지우기 | Tensor 뒤에 적용 |
| `Normalize` | 정규화 | 평균/표준편차 기준 변환 |
| `DataLoader` | batch 공급 도구 | `DataLoader(dataset)` |
| `Fine-tuning` | 전체 파라미터 미세 조정 | `net.parameters()` 학습 |
| `Transfer Learning` | feature extractor 동결 | classifier만 학습 |
| `requires_grad` | gradient 계산 여부 | `False`면 동결 |
| `VGG19-BN` | VGG19 + BatchNorm | pretrained model |
| `classifier[6]` | VGG19-BN 마지막 layer | 출력 class 수 교체 |
| `ResNet18` | 잔차 연결 기반 CNN | pretrained model |
| `fc` | ResNet18 마지막 layer | 출력 class 수 교체 |
| `CosineAnnealingLR` | 학습률 scheduler | cosine 형태로 감소 |
| `EarlyStopping` | 조기 종료 | val loss 기준 |
| `Confusion Matrix` | 예측 오류표 | class별 혼동 확인 |
| `classification_report` | class별 평가 리포트 | precision, recall, f1 |
| `tqdm` | 진행률 표시 | 학습 진행 상황 표시 |

---

## 7. 그래프/출력 결과 해석

### 7.1 이미지 미리보기

`ImageFolder`가 이미지를 제대로 읽었는지, label이 맞게 붙었는지 확인한다.

### 7.2 Loss Curve

train loss와 val loss가 함께 내려가면 학습이 안정적이다.

train loss만 내려가고 val loss가 올라가면 과적합을 의심한다.

### 7.3 Accuracy Curve

train accuracy와 val accuracy가 함께 올라가면 일반화가 잘 되는 흐름이다.

val accuracy가 train accuracy보다 높게 나올 수도 있는데, 데이터가 적거나 augmentation 때문에 train이 더 어렵게 구성된 경우 생길 수 있다.

### 7.4 Learning Rate Schedule

CosineAnnealingLR을 쓰면 learning rate가 점점 작아진다.

후반에는 큰 변화보다 미세 조정을 하도록 만든다.

### 7.5 Confusion Matrix

대각선은 맞힌 개수다.

대각선 밖은 모델이 어떤 class를 다른 class로 착각했는지 보여준다.

### 7.6 샘플 예측 시각화

정답과 예측을 함께 보며 모델이 실제 이미지에서 어떤 오류를 내는지 확인한다.

---

## 8. 실습에서 배운 점

- 실제 프로젝트에서는 사용자가 직접 준비한 이미지 폴더를 다루는 일이 많다.
- `ImageFolder`를 쓰면 폴더 구조만 맞춰도 Dataset을 쉽게 만들 수 있다.
- 학습 데이터와 검증 데이터의 transform은 다르게 구성해야 한다.
- 학습 데이터에는 증강을 넣어 과적합을 줄인다.
- 검증 데이터에는 랜덤 증강을 넣지 않는다.
- VGG19-BN에서는 `classifier[6]`을 바꿔야 한다.
- ResNet18에서는 `fc`를 바꿔야 한다.
- Fine-tuning은 전체 모델을 학습한다.
- Transfer Learning은 대부분의 가중치를 동결하고 마지막 분류기만 학습한다.
- 데이터가 적을수록 transfer learning이 안전한 경우가 많다.
- Early Stopping과 Scheduler는 실전 학습 루프에서 자주 쓰인다.
- Confusion Matrix와 Classification Report는 전체 정확도보다 자세한 평가를 가능하게 한다.

---

## 9. 시험용 요약

```text
15강 핵심 = 직접 준비한 이미지 폴더를 ImageFolder로 Dataset으로 만들고 사전학습 모델에 연결하는 것
```

- `ImageFolder`는 폴더 이름을 label로 인식한다.
- 폴더 구조는 `train/class_name/image.jpg`, `val/class_name/image.jpg`가 기본이다.
- `class_to_idx`는 class 이름과 숫자 label 매핑이다.
- 학습 transform에는 랜덤 증강을 넣는다.
- 검증 transform에는 랜덤 증강을 넣지 않는다.
- `RandomErasing`은 `ToTensor()` 뒤에 와야 한다.
- `DataLoader`는 Dataset을 batch 단위로 공급한다.
- Fine-tuning은 전체 파라미터를 업데이트한다.
- Transfer Learning은 기존 가중치를 동결하고 마지막 분류기만 학습한다.
- `requires_grad=False`는 해당 parameter를 학습하지 않겠다는 뜻이다.
- VGG19-BN의 마지막 layer는 `classifier[6]`이다.
- ResNet18의 마지막 layer는 `fc`다.
- ResNet18은 `model.fc = nn.Linear(model.fc.in_features, 10)`으로 CIFAR-10에 맞춘다.
- `CosineAnnealingLR`은 learning rate를 cosine 곡선처럼 줄인다.
- Early Stopping은 검증 손실이 개선되지 않을 때 학습을 멈춘다.
- Confusion Matrix는 class별 오분류를 확인하는 표다.