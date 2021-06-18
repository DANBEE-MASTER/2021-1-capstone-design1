# X-ray 이미지 및 딥러닝을 이용한 병변 인식
#### - 팀원 &nbsp;소개&nbsp;&nbsp;&nbsp;&nbsp; : &nbsp;강태구, 박기범, 박장선, 손주혜
#### - 산학 멘토님 : &nbsp;송세헌 멘토님
#### - 지도 교수님 : &nbsp;이승규 교수님
<br/>
  
## 1. 배경  
최근 인공지능 기술이 급부상하면서 의료 영역 내에서도 적용 범위가 확대되어 가고 있다. 특히나 이미 딥러닝을 이용한 신경망은 의료영상 분석 분야에 적용하여 효율성이 증명되었다. 의료영상을 이해하는 면에서 최근의 기계 학습의 발전은 Deep Learning에서 의료영상의 패턴을 식별 및 분류 연구에 기여하고 있다. 의료영상 분석에 인공신경망을 기반으로 하는 CNN(Convolutional Neural Network) 알고리즘이 본격적으로 사용되기 시작하면서 다양한 질환 분석 연구 사례가 급증하고 있다. 이러한 연구 등으로 인공지능 기술을 영상의학 검사의 영상 분석에 도입하여, 이전의 의료진들이 이미지를 직접 보며 판단하는 아날로그적 방식에서, 인공지능의 도입으로 전문의를 대체 가능할 정도의 수준까지 이르고 있다. <br/>
하지만 긴급을 요하는 의료현장에서 Chest X-ray 이미지의 병변을 판단하기 위해 Deep learning network model을 설계 및 구현 하기에는 어려움이 많다. 정상 Chest X-ray 이미지와 비교해 특정 질병은 현저하게 Chest X-ray 이미지가 부족하기도 하고, Medical Image 특성상 병변에 따라 좌우 반전, 상하반전과 같은 Data augmentation을 사용할 수 없어, Deep learning network model에 유용한 Data augmentation 방법을 찾기도 쉽지 않다. 또한 데이터의 불균형으로 Deep learning network model의 Depth, 파라미터 튜닝에 따른 학습 성능 차이가 날 가능성이 높으며, 공개 및 검증된 데이터셋을 활용하여 만든 사전학습모델(Pre-trained model)이 다른 Chest X-ray 이미지에 좋은 결과를 낼 것이라고 예상하기도 어렵다. <br/>
위와 같은 문제를 인식하여 검증된 NIH Chest X-ray 데이터를 통해 병변 판단에 유용한 Data augmentation 방법을 찾고, 여러 Deep learning network model을 비교 및 검증하여 가장 유용한 Depth, 파라미터를 모색하여 사전학습모델(Pre-trained model)을 만들고, 만들어진 사전학습모델(Pre-trained model)을 Fine-tuning 하여 로컬 Chest X-ray 이미지의 심장비대증에 대한 병변 판단 binary classification 연구를 수행하고자 한다. <br/>


## 2.목표  
Deep learning network model에 NIH Dataset을 학습시키고 14개의 병증에 대한 Multi-classification 을 성공시킨다. Multi-classification에 한계가 발견될 시, 특정 병증에 대해서 Positive labelling을 하고, 이외의 모든 상태에 대해 Negative labelling을 적용함으로써 필터링 할 수 있는 것을 목표로 한다.<br/>
위 프로젝트를 바탕으로 X-ray 이미지에 적합한 데이터 증강(Data augmentation) 방법과 딥러닝 네트워크 모델(Deep learning network model)을 알 수 있으며, 사전학습모델(Pre-trained model)의 Imagenet 가중치 사용 여부에 따른 로컬 Chest X-ray 이미지의 병변 판단 성능을 보여줌으로써, 최적의 Data augmentation, Deep learning network model 방법을 제시하는 것을 목적으로 한다. 이로써 긴급을 요하는 의료현장에서 병변판단을 위한 물리적인 시간 문제 해결 뿐 아니라 응급상황을 선별하는데에 적용하거나 1차적 진단 도구로 활용될 수 있으며 향후 흉부 방사선 자동판독기술의 고도화 연구로 연결될 수 있을 것으로 기대된다.<br/>


## 3. 프로젝트 활동 상세 내용 
#### 1) Data augmentation
Data augmentation 방법에 따른 성능 차이를 살펴보기 위해, X-ray 이미지 정상 2000장, 심장비대증 1000 → 2000장으로 이미지 증대 시킨 이후, DenseNet121 네트워크 모델을 사용하여 학습시킨 이후 Test set에 대한 accuracy 와 클래스에 따른 ROC curve와 AUC 값을 확인하였다. <br/>
 <br/>
![](http://khuhub.khu.ac.kr/2021-1-capstone-design1/LSK_Project1/raw/master/README_IMG/01.PNG)
다음은 가장 성능이 안좋았던 이미지증대 방법인 가우시안 블러, 가장 성능이 좋았던 이미지증대 방법인 ~5% 랜덤 회전에 대한 Feature Map 추출과 그에 따른 Grad-Cam이다. Grad-Cam을 확인해 보면 같은 이미지라도 네트워크가 전혀 다른 곳을 바라보고 있다는 것을 확인할 수 있다. <br/>
 <br/>
![](http://khuhub.khu.ac.kr/2021-1-capstone-design1/LSK_Project1/raw/master/README_IMG/02.PNG)
