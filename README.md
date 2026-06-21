# AI Lecture Notes

두산 부트캠프 ROKEY 과정에서 학습한 **AI 개론 / Computer Vision / NLP** 내용을 차시별로 정리한 학습 저장소

---

## Repository Overview

| 항목 | 내용 |
| --- | --- |
| 과정 | 두산 부트캠프 ROKEY |
| 과목 | AI 개론 / Computer Vision 중심 실습 |
| 정리 범위 | Python, NumPy, Matplotlib, PyTorch, 머신러닝, 딥러닝, CNN, 전이학습, NLP |
| 정리 방식 | 강의자료 + 차시별 개념 README + 실행 가능한 Jupyter Notebook |
| 목적 | 수업 복습, 과제 정리, 코드 흐름 이해, 시험 전 요약, 포트폴리오 기록 |

---

## Folder Structure

```text
ai-lecture-notes-main/
├─ Assignment_For_Computer_Vision/
│  └─ 차시별 과제 제출 파일
│
├─ Each_Week_Summary/
│  └─ 주차별 요약 PDF / DOCX
│
├─ Day1/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day1_student_note_da.ipynb
│
├─ Day2/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day2_student_note_da.ipynb
│
├─ Day3/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day3_student_note_da.ipynb
│
├─ Day4/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day4_student_note_da.ipynb
│
├─ Day5/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day5_student_note_da.ipynb
│
├─ Day6/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day6_student_note_da.ipynb
│
├─ Day7/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day7_student_note_da.ipynb
│
├─ Day8/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day8_student_note_da.ipynb
│
├─ Day9/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  └─ Summary_Day9_student_note_da.ipynb
│
├─ Day10/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  ├─ Summary_Day10_online_student_note_da.ipynb
│  └─ Summary_Day10_offline_student_note_da.ipynb
│
├─ Day11/
│  ├─ Lecture_Materials/
│  ├─ README.md
│  ├─ Summary_Day11_student_note_da.ipynb
│  ├─ Summary_Day11_online_student_note_da.ipynb
│  └─ Summary_Day11_offline_student_note_da.ipynb
│
├─ Day12/
│  ├─ Lecture_Materials/
│  ├─ README_AI_12차시_튜닝_정리_student_note_da.md
│  ├─ Summary_Day12_tuning_student_note_da.ipynb
│  ├─ Summary_Day12_tuning_online_student_note_da.ipynb
│  └─ Summary_Day12_tuning_offline_student_note_da.ipynb
│
├─ Day13/
│  ├─ Lecture_Materials/
│  ├─ README_AI_13차시_정리_student_note_da.md
│  ├─ Summary_Day13_student_note_da.ipynb
│  ├─ Summary_Day13_online_student_note_da.ipynb
│  └─ Summary_Day13_offline_student_note_da.ipynb
│
├─ Day14/
│  ├─ Lecture_Materials/
│  ├─ README_AI_14차시_정리_student_note_da.md
│  ├─ Summary_Day14_student_note_da.ipynb
│  ├─ Summary_Day14_online_student_note_da.ipynb
│  └─ Summary_Day14_offline_student_note_da.ipynb
│
├─ Day15/
│  ├─ Lecture_Materials/
│  ├─ README_AI_15차시_정리_student_note_da.md
│  ├─ Summary_Day15_student_note_da.ipynb
│  ├─ Summary_Day15_online_student_note_da.ipynb
│  └─ Summary_Day15_offline_student_note_da.ipynb
│
├─ Day16/
│  ├─ Lecture_Materials/
│  ├─ README_AI_16차시_정리_student_note_da.md
│  ├─ Summary_Day16_student_note_da.ipynb
│  ├─ Summary_Day16_online_student_note_da.ipynb
│  └─ Summary_Day16_offline_student_note_da.ipynb
│
├─ Day17/
│  ├─ Lecture_Materials/
│  ├─ README_AI_17차시_정리_student_note_da.md
│  ├─ Summary_Day17_online_student_note_da.ipynb
│  └─ Summary_Day17_offline_student_note_da.ipynb
│
└─ README.md
```

---

## Learning Contents

| Day | Topic | Main Focus |
| --- | --- | --- |
| Day1 | Python 기초 / 실습 환경 구성 | 딥러닝 학습에 필요한 Python 문법, 개발 환경, 객체지향 기본 개념을 정리 |
| Day2 | NumPy / Matplotlib | 배열 연산, 행렬 처리, 데이터 시각화 |
| Day3 | PyTorch 기본 기능 | Tensor, 차원, 슬라이싱, 연산 흐름, 자동 미분의 기본 구조 |
| Day4 | AI 기초 / 신경망 | 신경망, 역전파, 활성화 함수, 데이터 분할과 누수 방지 개념 |
| Day5 | 경사하강법 / 모델 개발 | 예측 함수, Layer 함수, Gradient, Scheduler, BatchNorm |
| Day6 | 손실 함수 설계 | MSE, MAE, Huber, BCEWithLogits, CrossEntropy, Label Smoothing, 클래스 불균형 정리 |
| Day7 | 선형회귀 | OLS, `nn.Linear`, 단순회귀/중회귀, R² Score, 정규화 개념. |
| Day8 | 이진분류 | Sigmoid, BCE, BCEWithLogitsLoss, 평가 지표, 불균형 데이터, 실무형 분류 파이프라인 |
| Day9 | 다중분류 1 | Softmax, CrossEntropyLoss, `torch.max`, NLLLoss, KL Divergence, 다층 신경망 |
| Day10 | 다중분류 2 | MNIST 이미지 분류, MLP, Transform, Dataset/DataLoader, Mini-batch, GPU, 불균형 평가 |
| Day11 | CNN 기반 이미지 분류 | Conv2d, Kernel/Filter, Feature Map, Pooling, Flatten, CIFAR-10 기반 FCN vs CNN 구조. |
| Day12 | 튜닝 / 최적화 / 과적합 | Optimizer, Dropout, BatchNorm, Data Augmentation, Overfitting |
| Day13 | 사전학습 모델 활용 1 | Pretrained Model, Fine-Tuning, Transfer Learning, ResNet18, VGG19-BN, MobileNetV2 |
| Day14 | 사전학습 모델 활용 2 | Freeze, Partial Fine-tuning, Full Fine-tuning, 차등 학습률, 데이터 증강 전략 |
| Day15 | 사용자 데이터 분류 | Custom Dataset, ImageFolder, Transfer Learning, ResNet18 Fine-tuning 기반 사용자 데이터 분류 |
| Day16 | CNN 스크래치 구현 | SimpleCNN, SVHN, Hook, Residual Block, ResNet18 구조 감각 |
| Day17 | NLP 종합 실습 | BiLSTM, DistilBERT, Tokenizer, Transformer 기반 자연어처리 실습 흐름 |

---

## File Description

| File / Folder | Description |
| --- | --- |
| `Lecture_Materials/` | 강의 PDF, 원본 실습 노트북, 자막 파일, 보강 자료 |
| `README.md` | 핵심 개념, 코드 흐름, 함수 사용법, 헷갈리는 부분을 정리한 학습 노트 |
| `README_AI_N차시_정리_student_note_da.md` | 개념 정리 README |
| `Summary_DayN_student_note_da.ipynb` | Markdown 셀과 Code 셀을 분리한 실행 가능한 요약 노트 |
| `Summary_DayN_online_student_note_da.ipynb` | 온라인 환경에서 바로 실행하기 좋게 정리한 요약 노트 |
| `Summary_DayN_offline_student_note_da.ipynb` | 오프라인 환경 또는 로컬 실행을 고려해 정리한 요약 노트 |
| `Assignment_For_Computer_Vision/` | 차시별 과제 제출 파일과 주차별 과제 문서 |
| `Each_Week_Summary/` | 주차별 전체 요약 PDF/DOCX |

---

## Tools & Libraries

| Category | Tools / Libraries |
| --- | --- |
| Language | Python |
| Notebook | Jupyter Notebook |
| Data / Math | NumPy, pandas |
| Visualization | Matplotlib |
| Machine Learning | scikit-learn |
| Deep Learning | PyTorch, torchvision |
| Computer Vision | CNN, ResNet18, VGG, MobileNet, Transfer Learning |
| NLP | Tokenizer, RNN/LSTM, BiLSTM, Transformer, BERT, DistilBERT |

---

