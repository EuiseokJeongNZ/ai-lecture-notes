# 강의_9기_AI개론_14차시_정리  
## 사전학습 모델 활용 2 · 전이학습 전략 비교 · Freeze / Partial / Full Fine-tuning · 차등 학습률 · 데이터 증강

## 1. 개요

이번 14차시는 사전학습 모델을 어떻게 전략적으로 fine-tuning할지 다루는 강의다.

13차시에서는 사전학습 모델을 불러오고 마지막 분류기를 바꾸는 기본 흐름을 배웠다.  
14차시에서는 한 단계 더 들어가서 어떤 layer를 학습할지, 어떤 learning rate를 줄지, augmentation을 어느 정도 강하게 줄지 비교한다.

핵심 흐름은 다음이다.

```text
층별 동결 전략 비교
→ Full Freeze
→ Partial Fine-tuning
→ Full Fine-tuning
→ Overfit Gap 계산
→ 차등 학습률 Parameter Groups
→ Gradual Unfreezing
→ Weak / Medium / Strong Augmentation
→ 작은 데이터셋 전략 선택 가이드
```

이번 정리는 두 가지 실행 환경으로 나누었다.

```text
인터넷 가능 버전
→ CIFAR-10 다운로드와 ImageNet pretrained ResNet18을 사용한다.

인터넷 불가 버전
→ FakeData와 weights=None으로 구조를 연습한다.
```

강의 원본 흐름은 거의 online 버전 기준이다.

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| Transfer Learning | 이미 학습된 모델을 새 문제에 맞게 활용하는 방법 |
| Fine-tuning | 사전학습 모델의 일부 또는 전체를 새 데이터에 맞게 다시 학습하는 방식 |
| Freeze | parameter를 동결해 학습하지 않게 하는 것 |
| Unfreeze | 동결된 parameter를 다시 학습 가능하게 푸는 것 |
| Full Freeze | backbone 전체를 동결하고 classifier만 학습하는 전략 |
| Partial Fine-tuning | 마지막 block 일부와 classifier를 학습하는 전략 |
| Full Fine-tuning | 전체 layer를 학습하는 전략 |
| Backbone | feature extractor 역할을 하는 모델 앞부분 |
| Head | 최종 class를 분류하는 classifier 부분 |
| ResNet layer4 | ResNet의 마지막 block, partial 전략에서 자주 해제 |
| requires_grad | 해당 parameter의 gradient 계산 여부 |
| Overfit Gap | train accuracy와 validation/test accuracy 차이 |
| Differential Learning Rate | layer group마다 다른 learning rate를 주는 전략 |
| Parameter Groups | optimizer에 layer group별 설정을 넘기는 방식 |
| Gradual Unfreezing | head부터 시작해 점진적으로 layer를 푸는 전략 |
| Weak Augmentation | 좌우 반전 정도의 약한 증강 |
| Medium Augmentation | crop, color jitter 등 중간 강도 증강 |
| Strong Augmentation | rotation, random erasing 등 강한 증강 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `Summary_Day14_online_student_note_da.ipynb` | 인터넷 가능 버전, CIFAR-10 + pretrained ResNet18 기반 |
| `Summary_Day14_offline_student_note_da.ipynb` | 인터넷 불가 버전, FakeData + weights=None 구조 연습 |
| `Summary_Day14_student_note_da.ipynb` | 두 버전 안내용 파일 |
| `README_AI_14차시_정리_student_note_da.md` | 14차시 개념과 코드 흐름 정리 |
| `강의_9기_AI개론_14차시_01_전이학습_실습_섹션별.ipynb` | freeze / partial / full 전략 섹션별 실습 |
| `강의_9기_AI개론_14차시_02_전이학습_실험.ipynb` | 전략 비교, 차등 학습률, augmentation 실험 |
| `강의_9기_AI개론_14차시_(사전_학습_모델_활용_2).pdf` | 파인튜닝 전략, 데이터 크기별 가이드, augmentation 가이드 |
| `14. 사전학습 모델 2.srt` | 강의 스크립트 |

---

## 4. 코드 흐름 요약

### 인터넷 가능 버전

```text
1. 라이브러리 import
2. device와 class 이름 설정
3. 전이학습 전략 3가지 개념 정리
4. ImageNet 기준 transform 정의
5. CIFAR-10 다운로드
6. class당 50개씩 뽑아 작은 데이터셋 구성
7. DataLoader 생성
8. 이미지 역정규화 후 샘플 시각화
9. ResNet18 pretrained 모델 불러오기
10. build_model(strategy) 함수 정의
11. freeze, partial, full 전략별 requires_grad 설정
12. 학습 가능한 parameter 수 확인
13. train_one_epoch 함수 정의
14. evaluate_model 함수 정의
15. make_optimizer 함수 정의
16. run_strategy 함수로 전략별 학습
17. train/test accuracy 그래프 비교
18. Overfit Gap 표 계산
19. 차등 학습률 parameter group 구성
20. Gradual Unfreezing 함수 정의
21. weak, medium, strong augmentation 정의
22. 증강 강도별 DataLoader 구성
23. 작은 데이터셋 전략 선택 가이드 정리
```

### 인터넷 불가 버전

```text
1. 라이브러리 import
2. FakeData로 3×224×224 이미지 생성
3. DataLoader 구성
4. ResNet18(weights=None) 구조 생성
5. build_model(strategy) 함수 정의
6. freeze, partial, full 전략별 학습 가능한 parameter 수 확인
7. 학습/평가 함수 정의
8. 전략별 짧은 학습 실행
9. 결과 비교 표 생성
10. 차등 학습률 parameter group 확인
11. Tensor 기반 weak/medium/strong augmentation 흉내
12. 작은 데이터셋 전략 선택 가이드 정리
```

---

## 5. 주요 코드 블록 설명

### 5.1 ImageNet transform

```python
transform_basic = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])
```

사전학습 ResNet18 입력에 맞게 이미지를 224×224로 맞추고 ImageNet 기준으로 정규화한다.

---

### 5.2 작은 데이터셋 만들기

```python
class_indices = {i: [] for i in range(num_classes)}

for idx, (_, label) in enumerate(full_train_dataset):
    if len(class_indices[label]) < samples_per_class:
        class_indices[label].append(idx)
```

각 class에서 지정한 개수만큼 index를 모은다.

작은 데이터셋에서 전이학습 전략 차이를 비교하기 위한 준비다.

---

### 5.3 Full Freeze

```python
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
```

backbone 전체를 동결하고 새로 만든 `fc`만 학습한다.

작은 데이터셋에서 빠르고 안전한 기본 전략이다.

---

### 5.4 Partial Fine-tuning

```python
for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_classes)
```

전체를 동결한 뒤 마지막 block인 `layer4`와 `fc`만 학습한다.

성능과 안정성의 균형점으로 자주 쓰인다.

---

### 5.5 Full Fine-tuning

```python
for param in model.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

전체 layer를 학습한다.

데이터가 많고 정규화가 충분할 때 고려할 수 있다.

---

### 5.6 Overfit Gap

```python
overfit_gap = train_acc - test_acc
```

train accuracy와 test accuracy 차이다.

gap이 크면 train 데이터에 과하게 맞았을 가능성이 있다.

---

### 5.7 차등 학습률

```python
optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 0.001},
    {"params": model.fc.parameters(), "lr": 0.01}
])
```

backbone 쪽은 작은 learning rate, 새 classifier는 큰 learning rate를 준다.

사전학습된 지식을 보존하면서 새 분류기는 빠르게 학습하기 위한 전략이다.

---

### 5.8 Gradual Unfreezing

```python
freeze_all(model)
unfreeze_module(model.fc)
unfreeze_module(model.layer4)
unfreeze_module(model.layer3)
```

처음에는 head만 학습하고, 이후 뒤쪽 layer부터 점진적으로 푸는 전략이다.

작은 데이터셋에서 안정적으로 학습 범위를 넓힐 때 사용한다.

---

### 5.9 Weak Augmentation

```python
transforms.Resize(224)
transforms.RandomHorizontalFlip()
```

가장 기본적이고 안전한 증강이다.

---

### 5.10 Medium Augmentation

```python
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
transforms.ColorJitter(brightness=0.2, contrast=0.2)
```

crop과 색상 변화를 추가해 일반화 성능을 높일 수 있다.

작은/중간 데이터셋에서 많이 비교해볼 만하다.

---

### 5.11 Strong Augmentation

```python
transforms.RandomRotation(degrees=15)
transforms.RandomErasing(p=0.5)
```

강한 변형을 준다.

데이터가 충분할 때는 도움이 될 수 있지만, 작은 데이터셋이나 정밀 도메인에서는 원본 의미를 깨뜨릴 수 있다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `freeze` | 동결 | `requires_grad=False` |
| `unfreeze` | 해제 | `requires_grad=True` |
| `requires_grad` | gradient 계산 여부 | 학습 대상 제어 |
| `Full Freeze` | backbone 전체 동결 | fc만 학습 |
| `Partial Fine-tuning` | 일부 block 학습 | layer4 + fc |
| `Full Fine-tuning` | 전체 layer 학습 | 과적합 주의 |
| `backbone` | feature extractor | ResNet의 conv/layer 부분 |
| `head` | classifier | ResNet의 fc |
| `layer4` | ResNet 마지막 block | partial에서 자주 학습 |
| `fc` | fully connected layer | 마지막 분류기 |
| `param_groups` | optimizer group | group별 lr 지정 |
| `differential LR` | 차등 학습률 | backbone 작게, head 크게 |
| `overfit gap` | train acc - test acc | 과적합 지표 |
| `augmentation` | 데이터 증강 | 입력 이미지 변형 |
| `RandomResizedCrop` | 랜덤 crop 후 resize | medium/strong 증강 |
| `ColorJitter` | 밝기/대비 변화 | 색상 증강 |
| `RandomErasing` | 일부 영역 지우기 | strong 증강 |
| `Gradual Unfreezing` | 점진적 해제 | head부터 layer를 해제 |
| `FakeData` | 가짜 데이터셋 | offline 구조 연습 |

---

## 7. 그래프/출력 결과 해석

### 7.1 Training Accuracy 비교

train accuracy가 빠르게 오르면 모델이 학습 데이터에 잘 맞고 있다는 뜻이다.

하지만 train만 높으면 과적합일 수 있다.

### 7.2 Test Accuracy 비교

test accuracy는 일반화 성능을 보는 지표다.

작은 데이터셋에서는 partial 전략이 freeze와 full 사이의 좋은 균형점이 될 수 있다.

### 7.3 Overfit Gap 표

`Train Acc - Test Acc`가 클수록 과적합 위험이 크다.

성능이 높아도 gap이 크면 좋은 모델이라고 보기 어렵다.

### 7.4 단일 학습률 vs 차등 학습률 그래프

차등 학습률은 새 classifier를 빠르게 학습시키면서 backbone은 천천히 조정한다.

사전학습된 특징을 보존하는 데 유리하다.

### 7.5 Augmentation 강도 비교 그래프

weak, medium, strong augmentation의 test accuracy와 overfit gap을 비교한다.

데이터가 적을 때는 medium 정도가 안전하고, strong은 데이터가 충분할 때 효과가 커질 수 있다.

---

## 8. 실습에서 배운 점

- 전이학습 전략은 데이터 크기와 도메인 유사도에 따라 달라진다.
- `requires_grad`는 어떤 layer를 학습할지 정하는 핵심 속성이다.
- Full Freeze는 빠르고 안전하지만 적응력이 낮을 수 있다.
- Partial Fine-tuning은 작은/중간 데이터셋에서 균형이 좋다.
- Full Fine-tuning은 데이터가 충분할 때 유리하지만 과적합 위험이 있다.
- Overfit Gap은 전략 비교에서 중요한 지표다.
- 차등 학습률은 backbone과 head를 다르게 학습시키는 실무적 방법이다.
- Gradual Unfreezing은 안정적으로 학습 범위를 넓히는 전략이다.
- 데이터 증강은 과적합을 줄이고 일반화를 높이는 데 도움을 준다.
- 증강은 도메인에 따라 조심스럽게 선택해야 한다.
- validation/test 데이터에는 랜덤 augmentation을 적용하지 않는다.

---

## 9. 시험용 요약

```text
14강 핵심 = 전이학습에서 freeze 범위, learning rate, augmentation 강도를 전략적으로 고르는 것
```

- Full Freeze는 backbone을 모두 동결하고 classifier만 학습한다.
- Partial Fine-tuning은 마지막 block과 classifier를 학습한다.
- Full Fine-tuning은 전체 layer를 학습한다.
- `requires_grad=False`는 학습하지 않겠다는 뜻이다.
- `requires_grad=True`는 학습하겠다는 뜻이다.
- 작은 데이터셋에서는 Full Freeze 또는 Partial부터 시작하는 것이 안전하다.
- 큰 데이터셋에서는 Full Fine-tuning도 고려할 수 있다.
- Overfit Gap은 `Train Acc - Test Acc`다.
- 차등 학습률은 backbone에는 작은 lr, head에는 큰 lr을 준다.
- optimizer parameter group으로 layer별 lr을 다르게 줄 수 있다.
- Gradual Unfreezing은 head부터 시작해 점진적으로 layer를 푸는 방법이다.
- Weak Augmentation은 안전하지만 다양성이 적다.
- Medium Augmentation은 작은/중간 데이터셋에서 많이 쓴다.
- Strong Augmentation은 데이터가 많을 때 효과적일 수 있지만 왜곡 위험이 있다.
- validation/test에는 랜덤 증강을 적용하지 않는다.