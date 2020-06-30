'''
문) bmi.csv 데이터셋을 이용하여 다음과 같이 ANN 모델을 생성하시오.
  <조건1>   
   - 1개의 은닉층을 갖는 ANN 분류기
   - hidden nodes = 4
   - Hidden layer : relu()함수 이용  
   - Output layer : sigmoid()함수 이용 
     
  <조건2> hyper parameters
    최적화 알고리즘 : AdamOptimizer
    learning_rate = 0.1 ~ 0.01
    반복학습 : 300 ~ 500
'''

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.0 사용안함
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import OneHotEncoder, minmax_scale # y data 
from sklearn.metrics import accuracy_score # model 평가

import numpy as np
import pandas as pd
 
bmi = pd.read_csv('C:/ITWILL/6_Tensorflow/data/bmi.csv')
print(bmi.info())

# label에서 normal, fat 추출 
bmi = bmi[bmi.label.isin(['normal','fat'])]
print(bmi.head())

# 칼럼 추출 
col = list(bmi.columns)
print(col) 

# x,y 변수 추출 
x_data = bmi[col[:2]] # x변수
y_data = bmi[col[2]] # y변수

# y변수(label) 로짓 변환 dict
map_data = {'normal': 0,'fat' : 1}
y_data = y_data.map(map_data) # dict mapping

# x_data 정규화 함수 
def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

x_data = x_data.apply(normalize)

# numpy 객체 변환 
x_data = np.array(x_data)
y_data = np.transpose(np.array([y_data]))# (1, 15102) -> (15102, 1)

print(x_data.shape) # (15102, 2)
print(y_data.shape) # (15102, 1)


# x,y 변수 정의 
X = tf.placeholder(tf.float32, shape=[ None, 2]) # x 데이터 수
Y = tf.placeholder(tf.float32, shape=[None, 1]) # y 데이터 수 

tf.set_random_seed(1234)

##############################
### ANN network  
##############################


hidden_node = 4

#hidden layer 
w1 = tf.Variable(tf.random_normal([2,hidden_node])) #[input, output]
b1 = tf.Variable(tf.random_normal([hidden_node])) #[output]

#output layer
w2 = tf.Variable(tf.random_normal([hidden_node,1]))  #[input, output]
b2 = tf.Variable(tf.random_normal([1]))  #[output]

# 4. softmax 분류기 
# 1) 회귀방정식 : 예측치 
hidden_output = tf.nn.relu(tf.matmul(X, w1) + b1 )# 회귀모델 -> 활성함수 (relu)

#output layer의 결과
model = tf.matmul(hidden_output, w2) + b2

# sigmonoid(예측치)
sigmonoid = tf.sigmonoid(model) # 활성함수 적용(0~1) : y1:0.8,y2:0.1,y3:0.1

# (2) loss function : 

#1차 방법 : Cross Entropy 이용 : -sum(Y * log(model))  
#loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))

# 2차 방법 : Softmmax + CrossEntropy
loss = tf.reduce_mean(tf.nn.sigmonoid_cross_entropy_with_logits(
        labels = Y, logits = model))

# 3) optimizer : 오차(비용함수) 최소화(w, b update) 
train = tf.train.AdamOptimizer(0.01).minimize(loss) # 오차 최소화

# 4) cut off : 0.5

cut_off =tf.cast(sigmonoid>0.5, tf.float32())


# 5. model 학습 
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # w, b 초기화 
    
    feed_data = {X : x_data, Y : y_data}
    
    # 반복학습 : 500회 
    for step in range(500) :
        _, loss_val = sess.run([train, loss], feed_dict = feed_data)
        
        if (step+1) % 50 == 0 :
            print("setp = {}, loss = {}".format(step+1, loss_val))
    
    # model result
    y_pred_re = sess.run(Y, feed_dict = {X : x_data}) #예측치 
    y_true_re = sess.run(X, feed_dict = {Y : y_data}) # 정답

    #print("y pred =", y_pred_re)
    #print("y ture =", y_ture_re) 
       
    acc = accuracy_score(y_true_re, y_pred_re)
    print("accuracy =", acc) # accuracy = 0.98
    
    #print(y_pred_re)
    #print(y_true_re)
    
    import matplotlib.pyplot as plt 
    plt.plot(y_pred_re, color='r')
    plt.plot(y_true_re, color='b')
    plt.show()
 

















