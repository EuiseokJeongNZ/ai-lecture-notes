# 강의_9기_AI개론_12차시_정리  
## 튜닝/최적화 및 과적합 대응 · Optimizer · Dropout · BatchNorm · Data Augmentation

## 1. 개요

이번 12차시는 CNN 모델의 성능을 높이기 위한 튜닝과 과적합 대응 방법을 정리하는 강의다.

11차시까지는 CNN 구조를 만들고 이미지 분류를 수행했다.  
이번 강의에서는 같은 CNN 구조를 더 깊게 만들고, Optimizer를 바꾸고, Dropout과 BatchNorm과 Data Augmentation을 추가하면서 성능이 어떻게 달라지는지 본다.

핵심 흐름은 다음이다.

```text
신경망의 다층화
→ 최적화 함수
→ 과적합 이해
→ Dropout
→ 공통 함수의 라이브러리화
→ 깊은 CNN 모델 구현
→ Optimizer 비교
→ BatchNorm
→ Data Augmentation
→ train/eval mode
→ Softmax 기반 오답 분석
```

이번 정리는 두 가지 실행 환경으로 나누었다.

```text
인터넷 가능 버전
→ CIFAR-10을 torchvision.datasets로 다운로드해서 강의 흐름을 따라간다.

인터넷 불가 버전
→ 합성 CIFAR-10식 데이터를 직접 만들어 같은 튜닝 흐름을 실행한다.
```

---

## 2. 오늘 배운 핵심 개념

| 개념 | 필기식 정리 |
|---|---|
| Tuning | 모델 성능을 올리기 위해 설정과 구조를 조정하는 작업 |
| Overfitting | 훈련 데이터에 너무 맞아 검증 데이터 성능이 나빠지는 현상 |
| Deep CNN | Conv layer를 여러 층 쌓은 모델 |
| Optimizer | gradient를 보고 parameter를 어떻게 바꿀지 정하는 함수 |
| SGD | 가장 기본적인 최적화 함수 |
| Momentum | 이전 업데이트 방향을 반영하는 SGD 개선 방식 |
| Adam | Momentum과 adaptive learning rate 계열 장점을 결합한 Optimizer |
| Dropout | 학습 중 일부 뉴런을 무작위로 꺼 과도한 의존을 막는 기법 |
| Batch Normalization | mini-batch 단위로 feature 분포를 안정화하는 기법 |
| Data Augmentation | 이미지를 변형해 데이터 다양성을 늘리는 방법 |
| RandomHorizontalFlip | 이미지를 좌우 반전하는 증강 |
| RandomErasing | 이미지 일부 영역을 지우는 증강 |
| train mode | 학습 모드, Dropout과 BatchNorm이 학습용으로 동작 |
| eval mode | 평가 모드, Dropout은 꺼지고 BatchNorm은 running 통계를 사용 |
| Softmax | logits를 class별 확률로 변환하는 함수 |
| running_mean | BatchNorm이 훈련 중 누적하는 평균 |
| running_var | BatchNorm이 훈련 중 누적하는 분산 |

---

## 3. 실습 파일 구성

| 파일 | 내용 |
|---|---|
| `Summary_Day12_tuning_online_student_note_da.ipynb` | 인터넷 가능 버전, CIFAR-10 다운로드 기반 튜닝 실습 |
| `Summary_Day12_tuning_offline_student_note_da.ipynb` | 인터넷 불가 버전, 합성 이미지 데이터 기반 튜닝 실습 |
| `Summary_Day12_tuning_student_note_da.ipynb` | 두 버전 안내용 파일 |
| `README_AI_12차시_튜닝_정리_student_note_da.md` | 12차시 개념과 코드 흐름 정리 |
| `강의_9기_AI개론_12차시_01_dl_tuning.ipynb` | 원본 실습 노트북 |
| `강의_9기_AI개론_12차시_(튜닝_기법).pdf` | 수업 목표와 핵심 개념 PDF |
| `12. 튜닝-최적화 및 과적합(Overfitting).srt` | 강의 스크립트 |
| `(심화)경사하강법과_옵티마이저_구현하기_ok.ipynb` | Optimizer 내부 구현 심화 자료 |

---

## 4. 코드 흐름 요약

### 인터넷 가능 버전

```text
1. 라이브러리 import
2. device와 CIFAR-10 class 이름 설정
3. SGD 수동 업데이트 구조 확인
4. Momentum과 Adam Optimizer 생성 확인
5. Dropout train/eval 동작 확인
6. RandomHorizontalFlip, RandomErasing transform 정의
7. CIFAR-10 train/test 데이터 다운로드
8. 빠른 실행용 Subset과 DataLoader 구성
9. 이미지 출력 함수 정의
10. 공통 학습 함수 fit, evaluate_history, collect_predictions 정의
11. CNN_v2 깊은 모델 정의
12. CNN_v2 shape 확인
13. SGD, Momentum, Adam 비교 학습
14. Optimizer별 validation accuracy 비교
15. CNN_v3 Dropout 모델 정의
16. CNN_v4 BatchNorm 모델 정의
17. CNN_v4 + Adam + Data Augmentation 학습
18. 최종 예측 이미지 확인
19. Softmax로 class별 예측 확률 분석
20. train/eval mode 차이 정리
21. BatchNorm 내부 동작 확인
22. classification_report와 confusion_matrix 확인
```

### 인터넷 불가 버전

```text
1. 라이브러리 import
2. 합성 CIFAR-10식 이미지 데이터 생성
3. 이미지 시각화
4. train/test 분할
5. TensorDataset과 DataLoader 구성
6. SGD, Momentum, Adam 기본 확인
7. Dropout train/eval 동작 확인
8. Tensor 기반 데이터 증강 함수 정의
9. 공통 학습 함수 정의
10. CNN_v2 깊은 모델 정의
11. Optimizer 비교 학습
12. CNN_v3 Dropout 구조 정의
13. CNN_v4 BatchNorm 구조 정의
14. CNN_v4 + Adam + 증강 학습
15. 예측 이미지 확인
16. Softmax 확률 분석
17. BatchNorm train/eval 동작 확인
18. classification_report와 confusion_matrix 확인
```

---

## 5. 주요 코드 블록 설명

### 5.1 SGD 수동 업데이트

```python
W.data -= lr * W.grad.data
B.data -= lr * B.grad.data
```

gradient에 learning rate를 곱한 값을 빼서 parameter를 업데이트한다.  
실제 학습에서는 직접 `.data`를 수정하기보다 `optimizer.step()`을 사용한다.

---

### 5.2 Momentum

```python
optimizer = optim.SGD(net_params, lr=lr, momentum=0.9)
```

이전 업데이트 방향을 90% 정도 반영한다.  
깊은 모델에서 더 빠르고 안정적인 수렴을 기대할 수 있다.

---

### 5.3 Adam

```python
optimizer = optim.Adam(net.parameters())
```

Momentum 계열의 방향 정보와 adaptive learning rate 계열의 보폭 조절을 함께 사용하는 Optimizer다.  
기본 선택지로 자주 쓰인다.

---

### 5.4 Dropout 동작 확인

```python
dropout = nn.Dropout(0.5)

dropout.train()
outputs_train = dropout(inputs)

dropout.eval()
outputs_eval = dropout(inputs)
```

학습 모드에서는 일부 값이 0이 된다.  
평가 모드에서는 Dropout이 적용되지 않는다.

---

### 5.5 데이터 증강 Transform

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
```

좌우 반전과 일부 영역 지우기를 통해 모델이 더 다양한 입력을 보게 만든다.

---

### 5.6 CNN_v2 깊은 모델

```text
conv1-2: 32채널
conv3-4: 64채널
conv5-6: 128채널
pooling 3회
```

pooling으로 이미지 크기가 줄어드는 대신 channel 수를 늘려 더 많은 특징을 학습하도록 만든다.

---

### 5.7 padding=1

```python
nn.Conv2d(3, 32, 3, padding=1)
```

3×3 Conv에서 `padding=1`을 주면 입력의 H, W 크기를 유지할 수 있다.  
32×32 이미지는 Conv 후에도 32×32를 유지한다.

---

### 5.8 CNN_v3 Dropout 모델

```python
self.dropout1 = nn.Dropout(0.2)
self.dropout2 = nn.Dropout(0.3)
self.dropout3 = nn.Dropout(0.4)
```

네트워크가 깊어질수록 dropout 비율을 높인다.  
pooling 직후와 classifier 내부에 배치한다.

---

### 5.9 CNN_v4 BatchNorm 모델

```python
self.conv1, self.bn1, self.relu
self.conv2, self.bn2, self.relu
```

Conv 뒤, ReLU 앞에 BatchNorm을 넣는다.  
중요한 점은 채널 수가 같아도 같은 BatchNorm 인스턴스를 재사용하지 않는 것이다.

---

### 5.10 train mode와 eval mode

```python
net.train()
net.eval()
```

Dropout과 BatchNorm은 mode에 따라 다르게 동작한다.  
학습 루프 시작에는 `net.train()`을, 평가 루프 시작에는 `net.eval()`을 명시해야 한다.

---

### 5.11 Softmax 확률 분석

```python
probs = torch.softmax(output, dim=1)
probs_np = probs.data.to("cpu").numpy()[0]
```

모델 logits를 class별 확률로 바꾼다.  
오답 분석에서 어떤 class와 헷갈렸는지 확인할 수 있다.

---

### 5.12 BatchNorm 내부 상태

```python
bn.running_mean
bn.running_var
bn.weight
bn.bias
```

BatchNorm은 단순 함수가 아니라 내부 상태와 학습 파라미터를 가진 Layer다.  
훈련 중에는 mini-batch 통계를 사용하고 running 통계를 업데이트한다.  
평가 중에는 running 통계를 사용한다.

---

## 6. 주요 함수/변수/약어 정리

| 이름 | 뜻 | 사용법 |
|---|---|---|
| `tuning` | 성능 개선을 위한 조정 | optimizer, dropout, batchnorm 등 |
| `overfitting` | 과적합 | train 성능만 좋아지는 현상 |
| `optimizer` | 최적화 함수 | `optim.SGD`, `optim.Adam` |
| `SGD` | 기본 최적화 함수 | `optim.SGD(...)` |
| `momentum` | 이전 이동 방향 반영 | `momentum=0.9` |
| `Adam` | 안정적인 Optimizer | `optim.Adam(...)` |
| `Dropout` | 뉴런 일부 비활성화 | `nn.Dropout(p)` |
| `BatchNorm2d` | 2D feature map 정규화 | `nn.BatchNorm2d(channels)` |
| `Data Augmentation` | 데이터 변형 | flip, erasing 등 |
| `RandomHorizontalFlip` | 좌우 반전 | `p=0.5` |
| `RandomErasing` | 일부 영역 지우기 | `scale`, `ratio` 사용 |
| `train()` | 학습 모드 | Dropout 적용, BN 통계 업데이트 |
| `eval()` | 평가 모드 | Dropout 중단, BN running 통계 사용 |
| `running_mean` | BN 누적 평균 | eval에서 사용 |
| `running_var` | BN 누적 분산 | eval에서 사용 |
| `softmax` | logits를 확률로 변환 | `torch.softmax(output, dim=1)` |
| `padding` | Conv 가장자리 채우기 | 크기 유지에 사용 |
| `CIFAR-10` | 10개 class 컬러 이미지 데이터 | 3×32×32 |
| `TensorDataset` | Tensor 기반 Dataset | 인터넷 불가 버전에서 사용 |
| `DataLoader` | mini-batch 공급 | `DataLoader(...)` |

---

## 7. 그래프/출력 결과 해석

### 7.1 Optimizer 비교 그래프

SGD, Momentum, Adam의 validation accuracy를 비교한다.  
Momentum과 Adam은 깊은 모델에서 더 빠르게 성능이 오를 수 있다.

### 7.2 Dropout 출력

train mode에서는 일부 값이 0이 된다.  
eval mode에서는 입력이 그대로 통과한다.

### 7.3 데이터 증강 이미지

좌우 반전과 RandomErasing이 적용되면 원본과 조금 다른 이미지가 생긴다.  
모델은 매번 조금씩 다른 데이터를 보며 특정 특징에 과도하게 의존하지 않게 된다.

### 7.4 Loss/Accuracy 곡선

train loss와 validation loss가 함께 내려가면 정상 학습이다.  
train loss는 내려가는데 validation loss가 올라가면 과적합을 의심한다.

### 7.5 Confusion Matrix

대각선은 맞힌 class다.  
대각선 밖은 어떤 class를 어떤 class로 헷갈렸는지 보여준다.

### 7.6 Softmax 확률표

예측 확률표를 보면 모델이 오답을 냈을 때 어느 class와 헷갈렸는지 확인할 수 있다.

---

## 8. 실습에서 배운 점

- 깊은 CNN은 더 복잡한 특징을 학습할 수 있다.
- 하지만 깊다고 무조건 성능이 좋아지지는 않는다.
- Optimizer 선택은 성능과 수렴 속도에 영향을 준다.
- Momentum은 이전 업데이트 방향을 반영한다.
- Adam은 안정적인 기본 선택지로 자주 쓰인다.
- 과적합은 train 성능은 좋아지고 validation 성능은 나빠지는 현상이다.
- Dropout은 특정 뉴런에 과도하게 의존하는 것을 막는다.
- BatchNorm은 feature map 분포를 안정화한다.
- BatchNorm은 Conv 뒤, ReLU 앞에 두는 것이 표준적이다.
- BatchNorm 인스턴스는 각 Layer 위치마다 별도로 만들어야 한다.
- Data Augmentation은 데이터를 변형해 일반화 성능을 높인다.
- `net.train()`과 `net.eval()`을 명확히 구분해야 한다.
- Softmax 확률은 모델 오답 분석에 유용하다.

---

## 9. 시험용 요약

```text
12강 핵심 = 깊은 CNN의 성능을 높이기 위한 튜닝과 과적합 대응
```

- 신경망을 깊게 쌓으면 더 복잡한 특징을 학습할 수 있다.
- 하지만 깊다고 무조건 성능이 좋아지는 것은 아니다.
- Optimizer는 parameter를 어떻게 업데이트할지 정한다.
- SGD는 가장 기본적인 Optimizer다.
- Momentum은 이전 업데이트 방향을 반영한다.
- Adam은 안정적이고 빠른 수렴을 기대할 수 있다.
- 과적합은 train 성능은 좋아지는데 validation 성능이 나빠지는 현상이다.
- Dropout은 일부 뉴런을 무작위로 꺼서 과도한 의존을 막는다.
- Dropout은 train mode에서만 적용된다.
- BatchNorm은 mini-batch 기준으로 feature 분포를 정규화한다.
- BatchNorm은 Conv 뒤, ReLU 앞에 두는 것이 표준적이다.
- BatchNorm은 상태가 있는 Layer다.
- BatchNorm 인스턴스는 여러 위치에서 재사용하면 안 된다.
- Data Augmentation은 데이터를 변형해 다양성을 늘린다.
- RandomHorizontalFlip은 좌우 반전이다.
- RandomErasing은 이미지 일부 영역을 지운다.
- `net.train()`은 학습 모드다.
- `net.eval()`은 평가 모드다.
- Softmax 확률표는 모델이 어떤 class와 헷갈렸는지 분석할 때 사용한다.