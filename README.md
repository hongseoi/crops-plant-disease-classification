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


### 2. Image segemtation model 생성
1의 결과가 식물일 경우 맨앞의 잎만 남도록 배경을 제거하는 segmentation 수행
- dataset: 기존 식물 이미지 데이터
- model: U-Net
- output: segmented image


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
