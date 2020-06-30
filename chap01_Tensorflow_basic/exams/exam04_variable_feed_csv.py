'''
문3) bmi.csv 파일을 가져와서 1,2칼럼은 x변수에 3칼럼은 y변수에 저장하여 처리하시오.
   조건1> x변수 : placeholder()이용 None행2열 배열 선언
   조건2> y변수 : placeholder()이용 1차원 배열 선언
   조건3> 칼럼 단위 평균 계산, label 빈도수 출력   
    
<<출력 결과 예시>>
키 평균 : 164.938
몸무게 평균 : 62.41

label 빈도수 :
normal    7677
fat       7425
thin      4898 
'''

import pandas as pd 

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 
from sklearn.model_selection import train_test_split

bmi = pd.read_csv("c:/itwill/6_Tensorflow/data/bmi.csv")
print(bmi.info())

# x,y 공급 data 
x_data = bmi[['height', 'weight']] # 복수 칼럼 선택 
y_data = bmi['label'] # 단일 칼럼 선택 

# 변수 정의 
x = tf.placeholder(dtype=tf.float32, shape = [ None,2])
y = tf.placeholder(dtype=tf.float32, shape = [ None])    

# train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)
x_train.shape # (14000, 2)
y_train.shape #(14000,)

# session object 공급 
#훈련용
with tf.Session() as sess:
    feed_data = { x : x_train, y : y_train}
    x_val, y_val = sess.run([x,y], feed_dict = feed_data)
    print(x_val, y_val)  #10 20

#키와 몸무게 평균
    type(x_val)
    x_val.shape
    print("키 평균", x_val[:,0].mean()) #컬럼단위 평균
    print("몸무게 평균", y_val[:,1].mean())

#빈도수 
    print("label빈도수")
    y_Ser = pd.Series(Y_val)
    print(y_Ser.value_counts())











