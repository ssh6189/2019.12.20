머신러닝
기계(컴퓨터 알고리즘 스스로 학습하여 서로 다른 변수 간의 관계를 찾아 나가는 과정)

지도학습 : 정답을 알고 있는 상태에서 학습을 시킨다.
전처리작업(데이터 수집 분석) -> 모형 학습(훈련 데이터) -> 예측(검증 데이터)
-> 모형평가 -> 모형활용

모형 평가 방법이 다양한 편

알고리즘
(분석모형) : 회귀분석, 분류

비지도학습
군집분석

정답이 없는 상태에서 서로 비슷한 데이터를 찾아서 그룹화
모형평가방법이 제한적

지도학습 : 레이블된 데이터, 직접 피드백, 출력 및 미래 예측

비지도학습 : 레이블 및 타깃 없음, 데이터에서 숨겨진 구조 찾기, 피드백 없음

강화학습 : 결정과정, 보상시스템, 연속된 행동에서 학습

지도학습의 주요목적은 레이블된 훈련 데이터에서 모델을 학습하여 본적 없는
미래 데이터에 대해 예측을 만드는것

분류 : 스팸메일을 필터링 : 개별 클래스 레이블이 있는 지도 학습

회귀 : 예측변수(또는 설명 변수), 입력과 연속적인 반응변수가 주어졌을때 출력값을 예측하는
두 변수 사이의 관계를 찾습니다.

즉, x와 y가 주어졌을때, 샘플과 직선사의 거리가 최소가 되는 직선을 긋는것
일반적으로 평균 제곱거리를 사용

x:선형변수

쉽게 말해, x와 y의 관계를 찾는것

강화학습

강화학습은 환경과 상호 작용하여 시스템 성능을 향상하는것이 목적이다. 환경의 현재
상태는 정보는 보상신호를 포함하기 때문에, 강화학습을 지도학습의 일부로 생각할 수 있다.

비지도 학습 - 레이블되지 않거나 구조를 알 수 없는 데이터를 다룬다.

비지도 학습 기법을 사용하면 알려진 출력값이나 보상 함수의 도움을 받지 않고 의미 있는 정보를 추출
하기 이해 데이터 구조를 탐색할 수 있다.

군집 : 사전 정보 없이 쌓여있는 그룹 정보를 의미 있는 서브그룹 또는 클러스터로 조직하는 탐색적 데이터 분석 기법
클러스터링은 정보를 조직화하고 데이터에서 의미 있는 관계를 유도하는 도구

차원축소 : 잡음 데이터를 제거하기 위해 특성 전처리 단계에서 적용하는 방법, 관련 있는 정보를 대부분 유지하면서, 더 작은 차원의 부분공간으로 데이터 압축
차원축소는 데이터 시각화에 유용

하이퍼파라미터 : 모델 성능을 향상하기 위해 사용하는 다이얼
모델 성능을 상세하기 조정하기 위해 주로 사용



우리는 주어진 문제에서 어떤 방법을 써야하는지, 기본개념 이런것 위주로 알면 된다.

회귀분석
가격, 매출, 주가, 환율, 수량 등 연속적인 값을 갖는 연속 변수를 예측하는데 주로 활용
분석 모형이 예측하고자 하는 목표를 종속변수 또는 예측 변수라고 한다.

예측을 위해 모형이 사용하는 속성을 독립변수 또는 설명변수라고 부른다.

seaborn 라이브러리의 regplot()함수를 이용하여 두 변수에 대한 산점도를 그린다.
기본적으로 회귀선을 표시한다. (fit_reg=False옵션을 회귀선으로 제거)

Seaborn 라이브러리의 jointplot(kind='reg'):회귀선을 표시한다.

Seaborn 라이브러리의 pairplot(kind='reg'):데이터 프레임의 열두개씩 짝을 지을 수 있는
모든 경우의 수에 대하여 두 변수간의 산점도를 그린다.

sklearn 라이브러리에서 선형회귀 분석 모듈을 사용한다.

LinearRegression() : 회귀분석 모형 객체를 생성
회귀분석모형객체.fit() : 회귀 방정식의 계수 a, b를 찾는다.
score() : 검증 데이터를 전달하여 회귀모형의 결정계수(R-제곱)을 구한다.
(결정계수값이 클수록 모형의 예측 능력이 좋다고 판단한다.)

모형객체의 coef_속성:기울기
모형객체의 intercept_속성 : 절편
predict():모형 객체의 예측값 반환
Seaborn라이브러리의 distplot():분포도를 그려 비교

단변량 선형 회귀는 하나의 특성(설명변수)와 연속적인 타깃(응답 변수) 사이의 관계를 모델링합니다.
특성이 하나인 선혀모델 공식은 다음과 같습니다.

다항회귀분석

두변수간의 관계를 보다 복잡한 고선 형태의 회귀선으로 표현할 수 있다.
2차 이상의 다항함수를 잉ㅇ하여 두 변수간의 선형 관계를 설명하는 알고리즘
PolynominalFeature(degree, ):다항식 반환
fit_trasform() 2차항 회귀분석에 맞게 변환
fit() - 모형 학습시킨다.
score() - 모형의 결정계수(R-제곱)을 구한다.
predict() - 검정 데이터를 입력하여 모형의 예측값 반환

y = w0 + w1x + w2x^2

PolynominalFeatures(degree=.): 다항식 변환
fit_transform() - 2차항 회귀분석에 맞게 변환
fit() - 모형학습시킨다.
score() - 모형의 결정계수(R-제곱)을 구한다.
predict) - 검정 데이터를 입력하여, 모형의 예측값 반환

다중회귀분석 (Polynominal Regression)
여러개의 독립변수가 종속 변수에 영향을 주고 선형 관계를 갖는 경우에 사용
설명변수(독립변수)가 2개 이상인 회귀모형을 분석대상으로한다.
추가적인 독립변수를 도입함으로써 오차항의 값을 줄일 수 있다.
종속변수를 설명하는 독립변수가 두개일때 단순회귀모형을 설정한다면, 모형설정이 부정확할 뿐 아니라
종속변수에 대한 중요한 설명변수(독립변수)를 누락함으로써 계수 추정량에 대해 편의를 야기시킬 수 있으므로

로지스틱 회귀분석
로지스틱 회귀분석은 종속변수와 독립변수간의 관계를 나타내어 예측모델을 생성한다는 점에서 선형회귀
분석과 비슷하지만, 종속변수의 결과가 범주형으로 분류분석에 해당된다.

연구내용 : 연봉, 야근횟수, 복지만족도, 업무 적합도가 퇴사에 미치는 영향

연속형자료(연봉, 야근횟수, 복지만조도, 업무적합도)가 범주형자료(퇴사한다, 안한다)에 미치는 영향을 분류한다.
로지스틱회귓분석은 종속변수(Y)에 로지스틱변환을 실시하여 로지스틱 회귀분석이라고 한다.
로지스틱 모형식은 독립변수(X) 값에 관계없이 종속변수의 값이 항상 0 - 1사이에 있도록 한다.

LogisticRegression

p-value값은 정규성검증을 위해 사용되는 수치이다. p값은 아무런 관련이 없는데, 유의미하게 나올 확률을 뜻한다.
즉, P값이 0.01이라면, 관련이 없는데 유의미하게 나올 확률이 1프로로 합리적인 변수로 판단할 수 있다.

귀납가설, 영가설 : ~차이가 없다.(차이가 없다, 영향이 없다.)
연구가설(대립가설) -> (차이가 있다. 영향이 있다.)

유이수준 - 0.05미만이면, 영향을 주지 않는다.귀납가설채택, 0.05이상이면, 영향을 준다. 연구가설 채택

-0.01

