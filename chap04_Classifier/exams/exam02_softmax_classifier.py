'''
문) bmi.csv 데이터셋을 이용하여 다음과 같이 softmax classifier 모델을 생성하시오. 
   조건1> bmi.csv 데이터셋 
       -> x변수 : height, weight 칼럼 
       -> y변수 : label(3개 범주) 칼럼
    조건2> 5,000번 학습, 500 step 단위로 loss, Accuracy 출력 
    조건3> 분류정확도(Accuracy) 출력  
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
from sklearn.preprocessing import minmax_scale # x data 정규화(0~1)
from sklearn.preprocessing import OneHotEncoder # y data -> one hot
import numpy as np
import pandas as pd
 
bmi = pd.read_csv('c:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# 칼럼 추출 
col = list(bmi.columns)
print(col) 

# x,y 변수 추출 
x_data = bmi[col[:2]] # x변수

# x_data 정규화 
x_data = minmax_scale(x_data)


# label one hot encoding 
label_map = {"thin": [1,0,0], "normal": [0,1,0], "fat": [0,0,1]}
bmi["label"] = bmi["label"].apply(lambda x : np.array(label_map[x]))

y_data = list(bmi["label"]) # 중첩list : [[1,0,0], [1,0,0]]
print(y_data[:5])

# numpy 객체 변환 
x_data = np.array(x_data)
y_data = np.array(y_data) 

print(x_data.shape) # (20000, 2)
print(y_data.shape) # (20000, 3)

# x, y변수 선언
X  = tf.placeholder(tf.float32, [None, 2]) # 키와 몸무게  
Y = tf.placeholder(tf.float32, [None, 3]) # 정답 레이블

# 가중치 : [입력층, 출력층)] 정의 
w = tf.Variable(tf.random_normal([2, 3]))
# 절편 : [출력수]
b = tf.Variable(tf.zeros([3]))

# 4. softmat 알고리즘(1~4)
# 1) model

# 2) cost : softmax + entropy

# 3) optimizer

# 4) predict vs real value


# 5. model training : <조건3> ~ <조건4>
with tf.Session() as sess :
    pass









