# crops-plant-disease-classification
동국대학교 2023-2 융합캡스톤디자인 민서두서 팀
ResNet 기반의 식물 질병진단 모델입니다.

---
## Full Process

### 1. 식물여부 이진분류 모델 생성
식물인지, 식물이 아닌지 판단하는 모델
- dataset: imageNet(mini)
- model: CNN
- output: binary class("plant", "non-plant")



### 3. fine-grained classification model
2의 결과물에 대해서 28개 라벨에 대해 진단하는 식물 질병 진단 모델
- dataset: segmentated 식물잎 이미지 데이터
- model: ResNet
- output: 28개 label

---

## Skills
| 역할 | 종류 |
|---|---|
| Language | Python |
| Framework | Pytorch |
| Version control | Git, GitHub | 

---
## 파일 구조
"""
/
├── 📂model
│   └── 🧾model.py # 예전 모델 코드
│   └── 🧾S1_Binary.ipynb # 식물 비식물 이진분류 모델
│   └── 🧾S3_classificaiton.ipynb # 식물 질병분류 모델
├── 📂output
│   └── 🧾**.pth # 모델 훈련 결과 파라미터값
│   └── 🧾**training_history.png # 모델 훈련 과정 history
├── 📂preprocess #이미지 전처리 코드
│   └── 🧾rename.ipynb 
│   └── 🧾resize.ipynb 
├── 📂sagemaker
├── 📂torchserver
├── 🧾label_data.csv # 프론트엔드 ui 구현 위한 정보 데이터
├── 🧾requirements.txt
└── 🧾README.md
"""