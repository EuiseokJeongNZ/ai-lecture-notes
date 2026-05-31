# Computer Vision

두산 부트캠프 ROKEY 과정에서 학습한 **Computer Vision / AI 개론** 내용을 정리한 저장소다.

각 차시별로 원본 강의자료, 개념 정리 README, 실행 가능한 Summary Notebook을 함께 관리한다.

---

## Repository Overview

* 과목: Computer Vision
* 정리 범위: AI 기초, Python, NumPy, Matplotlib, PyTorch, 머신러닝, 딥러닝, CNN, 모델 튜닝
* 정리 방식: 강의자료 + 개념 README + 실행 가능한 Jupyter Notebook
* 목적: 수업 복습, 과제 정리, 코드 흐름 이해, 포트폴리오 기록

---

## Folder Structure

```text
Computer_Vision/
├─ Assignment_For_Computer_Vision/
│  └─ 차시별 과제 제출 파일
│
├─ Each_Week_Summary/
│  └─ 주차별 요약 PDF
│
├─ Day1/
│  ├─ Lecture_Materials/
│  ├─ README_AI_1차시_정리_student_note_da.md
│  └─ Summary_Day1_student_note_da.ipynb
│
├─ Day2/
│  ├─ Lecture_Materials/
│  ├─ README_AI_2차시_정리_student_note_da.md
│  └─ Summary_Day2_student_note_da.ipynb
│
├─ Day3/
│  ├─ Lecture_Materials/
│  ├─ README_AI_3차시_정리_student_note_da.md
│  └─ Summary_Day3_student_note_da.ipynb
│
├─ Day4/
│  ├─ Lecture_Materials/
│  ├─ README_AI_4차시_정리_student_note_da.md
│  └─ Summary_Day4_student_note_da.ipynb
│
├─ Day5/
│  ├─ Lecture_Materials/
│  ├─ README_AI_5차시_정리_student_note_da.md
│  └─ Summary_Day5_student_note_da.ipynb
│
├─ Day6/
│  ├─ Lecture_Materials/
│  ├─ README_AI_6차시_정리_student_note_da.md
│  └─ Summary_Day6_student_note_da.ipynb
│
├─ Day7/
│  ├─ Lecture_Materials/
│  ├─ README_AI_7차시_정리_student_note_da.md
│  └─ Summary_Day7_student_note_da.ipynb
│
├─ Day8/
│  ├─ Lecture_Materials/
│  ├─ README_AI_8차시_정리_student_note_da.md
│  └─ Summary_Day8_student_note_da.ipynb
│
├─ Day9/
│  ├─ Lecture_Materials/
│  ├─ README_AI_9차시_정리_student_note_da.md
│  └─ Summary_Day9_student_note_da.ipynb
│
├─ Day10/
│  ├─ Lecture_Materials/
│  ├─ README_AI_10차시_정리_student_note_da.md
│  ├─ Summary_Day10_online_student_note_da.ipynb
│  └─ Summary_Day10_offline_student_note_da.ipynb
│
├─ Day11/
│  ├─ Lecture_Materials/
│  ├─ README_AI_11차시_정리_student_note_da.md
│  ├─ Summary_Day11_student_note_da.ipynb
│  ├─ Summary_Day11_online_student_note_da.ipynb
│  └─ Summary_Day11_offline_student_note_da.ipynb
│
├─ Day12/
│  ├─ Lecture_Materials/
│  ├─ README_AI_12차시_정리_student_note_da.md
│  └─ Summary_Day12_student_note_da.ipynb
│
└─ README.md
```

---

## Learning Contents

| Day   | Topic              | Main Focus                                          |
| ----- | ------------------ | --------------------------------------------------- |
| Day1  | Python 기초          | 딥러닝 학습에 필요한 Python 기본 문법을 정리한다.                     |
| Day2  | NumPy / Matplotlib | 배열 연산, 시각화, 데이터 처리의 기본 흐름을 학습한다.                    |
| Day3  | PyTorch 기본 기능      | Tensor, 차원, 슬라이싱, 연산 흐름을 학습한다.                      |
| Day4  | 신경망 기초             | 신경망, 역전파, 활성화 함수, 데이터 분할 개념을 정리한다.                  |
| Day5  | 모델 개발과 경사하강법       | 모델 학습 흐름, Gradient, Scheduler, BatchNorm을 학습한다.     |
| Day6  | 비용 함수와 손실 함수       | MSE, BCEWithLogits, Label Smoothing, 클래스 불균형을 정리한다. |
| Day7  | 선형회귀               | 회귀 모델과 머신러닝/딥러닝 기본 이론을 학습한다.                        |
| Day8  | 이진분류               | Binary Classification과 실무형 예제 흐름을 정리한다.             |
| Day9  | 다중분류 1             | Multi-class Classification의 기본 구조를 학습한다.            |
| Day10 | 다중분류 2             | 실전 다중분류 프로젝트, 평가지표, 클래스 불균형 처리를 정리한다.               |
| Day11 | CNN 기반 이미지 분류      | CNN 구조와 이미지 분류 모델의 기본 흐름을 학습한다.                     |
| Day12 | 튜닝 / 최적화 / 과적합     | Optimizer, Overfitting, 모델 튜닝 기법을 정리한다.             |

---

## Study Flow

각 차시는 아래 순서로 복습한다.

1. `Lecture_Materials/`에서 원본 강의자료를 확인한다.
2. `README_AI_N차시_정리_student_note_da.md`에서 개념과 코드 흐름을 먼저 읽는다.
3. `Summary_DayN_student_note_da.ipynb`를 실행하면서 코드가 어떻게 동작하는지 확인한다.
4. 함수 사용법, 인자 의미, 출력 결과를 다시 정리한다.
5. 과제 파일과 주차별 요약본을 참고해 시험 전 복습을 진행한다.

---

## File Description

| File / Folder                         | Description                                      |
| ------------------------------------- | ------------------------------------------------ |
| `Lecture_Materials/`                  | 강의 PDF, 원본 실습 노트북, 자막 파일 등 원본 자료를 보관한다.          |
| `README_AI_N차시_정리_student_note_da.md` | 각 차시의 핵심 개념, 코드 흐름, 함수 사용법, 헷갈리는 부분을 정리한 학습 노트다. |
| `Summary_DayN_student_note_da.ipynb`  | Markdown 셀과 Code 셀을 분리한 실행 가능한 요약 노트북이다.         |
| `Assignment_For_Computer_Vision/`     | 차시별 과제 제출 파일을 보관한다.                              |
| `Each_Week_Summary/`                  | 주차별 전체 요약 PDF를 보관한다.                             |

---

## Tools & Libraries

이 저장소에서는 주로 아래 도구와 라이브러리를 사용한다.

* Python
* Jupyter Notebook
* NumPy
* Matplotlib
* PyTorch
* scikit-learn
* CNN / Deep Learning 관련 실습 코드

---

## Notes

* 이 저장소는 수업 내용을 복습하기 위한 개인 학습 기록이다.
* 각 README는 교재처럼 딱딱하게 정리하기보다, 실제 강의를 들으며 다시 보기 좋도록 학습 노트 형태로 작성한다.
* Summary Notebook은 설명 셀과 실행 코드 셀을 분리하여, 개념을 읽고 바로 코드를 실행할 수 있도록 구성한다.
* 코드에서는 함수의 기본 사용 형태, 인자의 의미, 실습에서 왜 필요한지를 함께 정리한다.
* 차시별 내용은 학습이 진행되면서 계속 업데이트한다.
