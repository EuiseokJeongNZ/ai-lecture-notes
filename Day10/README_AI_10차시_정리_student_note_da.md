# 강의_9기_AI개론_10차시_정리  
## 다중 분류 2 · MNIST 이미지 분류 · MLP · Transform · Dataset/DataLoader · Mini-batch · GPU · 불균형 평가

## 1. 개요

이번 10차시는 9차시에서 배운 다중 분류를 이미지 데이터에 적용하는 강의다.

강사님은 오늘 수업을 CNN으로 넘어가기 전, 기본 MLP 딥러닝 구조를 마스터하는 과정이라고 설명했다.  
핵심 데이터는 MNIST 손글씨 숫자 이미지다.

이번 정리는 사용 환경을 두 가지로 나누었다.

```text
인터넷 가능 버전
→ torchvision.datasets.MNIST(download=True) 사용

인터넷 불가 버전
→ sklearn.load_digits() 내장 데이터 사용
```

두 버전의 핵심 흐름은 같다.

```text
이미지 = 픽셀 숫자 격자
이미지를 1차원 벡터로 Flatten
Dataset/DataLoader로 mini-batch 구성
MLP 모델 정의
CrossEntropyLoss로 다중 분류 학습
torch.max(outputs, 1)[1]로 예측 class 선택
classification_report와 confusion_matrix로 평가
```

강의에서 특히 강조된 부분은 이미지 Tensor의 구조다.  
일반 이미지나 NumPy에서는 HWC 순서를 자주 보지만, PyTorch에서는 CHW 순서를 많이 쓴다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| MNIST | 0~9 손글씨 숫자 이미지 데이터 |
| Pixel | 이미지를 구성하는 숫자 단위 |
| 28×28 | MNIST 한 장의 이미지 크기 |
| 784 | 28×28 이미지를 펼친 feature 수 |
| CHW | PyTorch 이미지 Tensor 순서, channel-height-width |
| HWC | 일반 이미지/NumPy에서 자주 보는 순서 |
| Flatten | 이미지를 1차원 벡터로 펼치는 과정 |
| Transform | 이미지 전처리 과정 |
| ToTensor | 이미지를 Tensor로 바꾸고 0~1 범위로 만드는 변환 |
| Normalize | 평균/표준편차 기준으로 값을 조정하는 변환 |
| Dataset | 데이터와 label을 묶은 객체 |
| DataLoader | mini-batch 단위로 데이터를 꺼내는 도구 |
| Mini-batch | 전체 데이터 중 일부 묶음 |
| GPU | 병렬 연산에 강한 장치 |
| device | Tensor와 모델이 올라가는 CPU/GPU 위치 |
| MLP | 완전 연결층 중심의 신경망 |
| ReLU | 은닉층에서 자주 쓰는 활성화 함수 |
| CrossEntropyLoss | 다중 분류 표준 손실함수 |
| torch.max | class별 점수 중 가장 큰 class를 찾는 함수 |
| macro/micro/weighted | 다중 분류 평균 지표 방식 |
| class weight | 소수 class를 더 크게 반영하는 가중치 |
| WeightedRandomSampler | 소수 class를 더 자주 뽑는 sampler |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `Summary_Day10_online_student_note_da.ipynb` | 인터넷 가능 버전, torchvision MNIST 다운로드 기반 실습 |
| `Summary_Day10_offline_student_note_da.ipynb` | 인터넷 불가 버전, sklearn digits 내장 데이터 기반 실습 |
| `README_AI_10차시_정리_student_note_da.md` | 10차시 개념과 코드 흐름 README |
| `10. 다중분류 2.srt` | 강사님 스크립트 기준 자료 |

---

## 4. 코드 흐름 요약

### 인터넷 가능 버전

```text
1. torch, torchvision, sklearn metric import
2. device 설정
3. Transform 정의
4. ToTensor로 0~1 Tensor 변환
5. Normalize로 -1~1 근처로 정규화
6. Lambda로 [1, 28, 28]을 [784]로 Flatten
7. datasets.MNIST(download=True)로 train/test 데이터 다운로드
8. Subset으로 빠른 실습용 데이터 구성
9. DataLoader로 mini-batch 생성
10. batch shape 확인
11. 이미지 하나를 다시 28×28로 시각화
12. MNISTMLP 모델 정의
13. CrossEntropyLoss와 Adam Optimizer 정의
14. 한 batch forward 확인
15. train_one_epoch 함수 정의
16. evaluate 함수 정의
17. epoch 반복 학습
18. loss/accuracy 그래프 출력
19. classification_report와 confusion_matrix 확인
20. softmax로 개별 예측 확률 확인
21. CNN용 4D 이미지 Tensor [batch, 1, 28, 28] 확인
```

### 인터넷 불가 버전

```text
1. sklearn.load_digits 데이터 불러오기
2. [N, 8, 8] 이미지와 [N, 64] Flatten feature 확인
3. 이미지 시각화
4. 픽셀값을 0~1 범위로 조정
5. train/test 분할
6. Tensor 변환, label은 LongTensor로 변환
7. TensorDataset과 DataLoader 구성
8. batch shape 확인
9. CNN용 [N, 1, 8, 8] 4D Tensor 확인
10. OfflineDigitsMLP 모델 정의
11. CrossEntropyLoss와 Adam 정의
12. train/evaluate 함수 정의
13. epoch 반복 학습
14. loss/accuracy 그래프 출력
15. classification_report와 confusion_matrix 확인
16. macro/micro/weighted 평균 지표 정리
17. label_binarize로 다중 분류 ROC-AUC 계산
18. class imbalance 데이터 생성
19. class weight 계산
20. CrossEntropyLoss(weight=...) 사용법 정리
21. WeightedRandomSampler로 oversampling 구조 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 인터넷 가능 버전 Transform

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.Lambda(lambda x: x.view(-1))
])
```

MNIST 이미지를 Tensor로 바꾸고, 정규화한 뒤, 784개 벡터로 펼친다.

완전 연결 신경망은 2차원 이미지를 그대로 받지 않고 1차원 feature 벡터를 받기 때문에 Flatten이 필요하다.

---

### 5.2 MNIST 다운로드

```python
datasets.MNIST(root="./data", train=True, download=True, transform=transform)
```

인터넷이 되는 환경에서 MNIST를 직접 내려받는다.

`train=True`는 훈련 데이터, `train=False`는 테스트 데이터다.

---

### 5.3 DataLoader

```python
DataLoader(train_dataset, batch_size=500, shuffle=True)
```

데이터를 mini-batch 단위로 꺼낸다.

train loader는 순서를 섞기 위해 `shuffle=True`를 사용하고, test loader는 보통 `shuffle=False`를 사용한다.

---

### 5.4 MLP 모델

```python
nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

MNIST MLP의 기본 구조다.

784개 픽셀 feature를 128개 은닉 표현으로 바꾸고, 마지막에 숫자 10개 class에 대한 logits를 출력한다.

---

### 5.5 CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()
```

다중 분류 표준 손실함수다.

모델 마지막에 Softmax를 붙이지 않고 logits를 그대로 넣는다.

---

### 5.6 예측 class 구하기

```python
pred = torch.max(outputs, 1)[1]
```

각 샘플의 class별 점수 중 가장 큰 위치를 예측 class로 사용한다.

`torch.max(outputs, 1)`은 values와 indices를 반환하고, `[1]`은 indices를 의미한다.

---

### 5.7 device 이동

```python
images = images.to(device)
labels = labels.to(device)
model = model.to(device)
```

모델과 데이터가 같은 device에 있어야 연산할 수 있다.

하나는 CPU, 하나는 GPU에 있으면 런타임 에러가 난다.

---

### 5.8 인터넷 불가 버전 데이터

```python
digits = load_digits()
X_images = digits.images
X_flat = digits.data
```

인터넷 없이 사용할 수 있는 손글씨 숫자 데이터다.

MNIST와 달리 8×8 이미지라서 Flatten하면 64개 feature가 된다.

---

### 5.9 TensorDataset

```python
train_dataset = TensorDataset(X_train_t, y_train_t)
```

이미 준비된 Tensor 입력과 label을 dataset 형태로 묶는다.

다운로드형 Dataset이 없을 때 자주 사용한다.

---

### 5.10 4D Tensor

```python
X_images_4d = torch.FloatTensor(X_images_scaled).unsqueeze(1)
```

CNN에서 사용할 수 있는 `[N, 1, 8, 8]` 형태를 만든다.

이번 실습은 MLP이지만, 다음 CNN 강의에서는 이 형태가 중요해진다.

---

### 5.11 Classification Report

```python
classification_report(y_true, y_pred)
```

class별 precision, recall, f1-score, support를 한 번에 보여준다.

다중 분류 성능을 class별로 확인할 때 유용하다.

---

### 5.12 Confusion Matrix

```python
confusion_matrix(y_true, y_pred)
```

정답 class와 예측 class의 관계를 표로 보여준다.

어떤 숫자를 어떤 숫자로 헷갈렸는지 확인할 수 있다.

---

### 5.13 다중 분류 ROC-AUC

```python
label_binarize(y_true, classes=classes)
roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
```

다중 분류에서 ROC-AUC를 계산하려면 label을 one-hot 형태로 바꿔야 한다.

class별로 one-vs-rest 방식으로 계산한다고 보면 된다.

---

### 5.14 Class Weight

```python
class_weights = n_samples / (n_classes * class_counts)
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
```

소수 class일수록 더 큰 가중치를 부여한다.

불균형 데이터에서 소수 class를 더 신경 쓰게 만들 수 있다.

---

### 5.15 WeightedRandomSampler

```python
WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
```

소수 class 샘플이 더 자주 뽑히도록 샘플링 가중치를 준다.

`replacement=True`는 같은 샘플을 중복해서 뽑을 수 있다는 뜻이다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `MNIST` | 손글씨 숫자 이미지 데이터 | `datasets.MNIST(...)` |
| `digits` | sklearn 내장 손글씨 숫자 데이터 | `load_digits()` |
| `pixel` | 이미지 한 칸의 숫자값 | 밝기 정보 |
| `CHW` | PyTorch 이미지 순서 | channel, height, width |
| `Flatten` | 이미지 펼치기 | `[1,28,28] → [784]` |
| `ToTensor` | 이미지 Tensor 변환 | 0~1 범위로 변환 |
| `Normalize` | 정규화 | 평균/표준편차 기준 변환 |
| `Lambda` | 간단한 사용자 변환 | `lambda x: x.view(-1)` |
| `Dataset` | 데이터 묶음 | 이미지와 label 제공 |
| `TensorDataset` | Tensor 기반 Dataset | `TensorDataset(X, y)` |
| `DataLoader` | mini-batch 생성 | `DataLoader(dataset, batch_size=...)` |
| `batch_size` | 한 번에 학습할 샘플 수 | 64, 500 등 |
| `shuffle` | 데이터 순서 섞기 | train은 True |
| `device` | 연산 장치 | CPU 또는 GPU |
| `cuda` | NVIDIA GPU 연산 환경 | `torch.cuda.is_available()` |
| `MLP` | 완전 연결 신경망 | Flatten된 벡터 입력 |
| `ReLU` | 활성화 함수 | 음수는 0, 양수는 그대로 |
| `CrossEntropyLoss` | 다중 분류 손실 | logits + long label |
| `torch.max` | 최댓값과 index 반환 | 예측 class 추출 |
| `macro` | class별 단순 평균 | 불균형 확인에 중요 |
| `micro` | 전체 샘플 기준 평균 | 전체 샘플 관점 |
| `weighted` | class 개수 반영 평균 | support 반영 |
| `class_weight` | class별 손실 가중치 | 소수 class 보정 |
| `WeightedRandomSampler` | 가중치 기반 샘플러 | oversampling 효과 |

---

## 7. 그래프/출력 결과 해석

### 7.1 이미지 시각화

Flatten된 벡터를 다시 28×28 또는 8×8로 바꾸면 사람이 볼 수 있는 숫자 이미지가 된다.

### 7.2 Loss Curve

Loss가 epoch가 지날수록 내려가면 모델이 정답 class에 더 높은 점수를 주도록 학습되고 있다는 뜻이다.

### 7.3 Accuracy Curve

Accuracy가 올라가면 숫자 분류를 더 많이 맞히는 것이다.

### 7.4 Confusion Matrix

대각선은 맞힌 개수다.

대각선 밖의 값은 모델이 헷갈린 숫자 조합이다.

### 7.5 Classification Report

precision, recall, f1-score를 class별로 확인할 수 있다.

불균형 데이터에서는 accuracy보다 macro f1, weighted f1도 같이 봐야 한다.

### 7.6 ROC Curve

다중 분류에서는 class별 one-vs-rest 방식으로 ROC Curve를 그린다.

micro 평균은 전체 샘플 기준이고, macro 평균은 class별 평균이다.

---

## 8. 실습에서 배운 점

- 이미지는 픽셀 숫자 격자다.
- MNIST는 28×28 흑백 이미지다.
- PyTorch는 이미지 Tensor에서 CHW 순서를 자주 쓴다.
- MLP는 이미지를 1차원 벡터로 펼쳐 입력받는다.
- 28×28 이미지는 784개 feature가 된다.
- 인터넷 가능 환경에서는 torchvision MNIST를 직접 다운로드할 수 있다.
- 인터넷 불가 환경에서는 sklearn digits로 같은 흐름을 연습할 수 있다.
- DataLoader는 mini-batch 학습을 쉽게 만든다.
- train loader는 보통 shuffle=True를 사용한다.
- 모델과 데이터는 같은 device에 있어야 한다.
- ReLU는 은닉층에 비선형성을 추가한다.
- CrossEntropyLoss는 다중 분류 표준 손실함수다.
- torch.max(outputs, 1)[1]은 예측 class를 구한다.
- classification_report는 class별 평가를 보여준다.
- Confusion Matrix는 어떤 class끼리 헷갈리는지 보여준다.
- class weight와 WeightedRandomSampler는 불균형 데이터 대응 방법이다.

---

## 9. 시험용 요약

```text
10강 핵심 = MNIST 이미지 다중 분류를 MLP로 구현하는 흐름
```

- 컴퓨터는 이미지를 픽셀 숫자로 본다.
- MNIST는 28×28 흑백 손글씨 숫자 데이터다.
- 28×28은 784개 feature다.
- PyTorch 이미지 Tensor는 보통 CHW 순서다.
- MLP는 Flatten된 1차원 벡터를 입력으로 받는다.
- `ToTensor()`는 이미지를 Tensor로 만들고 0~1 범위로 바꾼다.
- `Normalize()`는 평균과 표준편차 기준으로 값을 조정한다.
- `DataLoader`는 mini-batch를 만든다.
- `batch_size`는 한 번에 학습할 데이터 개수다.
- train은 `shuffle=True`, test는 보통 `shuffle=False`다.
- GPU를 쓰려면 model과 Tensor를 모두 같은 device로 보내야 한다.
- MLP 구조는 `Linear → ReLU → Linear`다.
- MNIST는 class가 10개라 마지막 출력도 10개다.
- `CrossEntropyLoss`에는 Softmax를 먼저 붙이지 않는다.
- 다중 분류 label은 `torch.long` 타입이어야 한다.
- 예측 class는 `torch.max(outputs, 1)[1]`로 구한다.
- 인터넷이 없으면 sklearn `load_digits()`로 비슷한 흐름을 연습할 수 있다.
- class imbalance에서는 accuracy만 보면 위험하다.
- macro/micro/weighted 평균 차이를 알아야 한다.
- class weight는 소수 class 손실을 크게 만든다.
- WeightedRandomSampler는 소수 class를 더 자주 뽑는 oversampling 방식이다.