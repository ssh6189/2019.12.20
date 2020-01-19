##############로지스틱  회귀분석 실습 ##########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1단계: 데이터 준비

score = [1,3,5,7,9,2,4,6,8,10,11,23,54,64,89,45,60,100, 30, 44, 40, 35, 60, 77, 80]
grade = [1.0, 2.0, 3.4, 4.1, 2.2, 3.4, 5.3, 3.4, 2.24, 4.4, 3.4, 3.6, 2.7, 3.54, 4.21, 1.24, 1.76, 2.64, 3.21, 4.0, 3.34, 4.5, 3.13, 2.7, 1.3]
_pass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

df = pd.DataFrame({"score":score, "_pass":_pass})

print(df.info())

df = pd.DataFrame( {"score":score, "grade": grade, "_pass":_pass})
print(df.info())

X=df[['score', 'grade']]
Y=df[['_pass']]

# train data 와 test data를 7:3 비율로 분리
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)   


from sklearn.linear_model import LogisticRegression 
logR = LogisticRegression() 
logR.fit(X_train, Y_train)   #학습

print('정확도: ' , logR.score(X_train, Y_train))
print('정확도: ' , logR.score(X_test, Y_test))

from sklearn.metrics import classification_report
y_predict = logR.predict(X_test)
print(classification_report(Y_test, y_predict ))   #실제 합격/불합격 테스트 데이터,  모형으로부터 예측된 합격/불합격 테스트 데이터 

#Confusion Matrix 기반 정확률, 지지율, F1계수, 재현율을 계산해서 보고서 반환
from sklearn.metrics import classification_report
y_predict = logR.predict(X_test)
print(classification_report(Y_test, y_predict)) #실제 합격/불합격 테스트 데이터, 모형으로부터 예측된 합격/불합격 테스트 데이터

#방법 2:
import statsmodels.api as sm
logit = sm.Logit(df['_pass'], X)  #로지스틱 회귀분석 실행
result = logit.fit()
print(result.summary2())
print(result.params)   # 종속변수에 영향을 미치는 정도 파악  
