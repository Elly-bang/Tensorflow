# -*- coding: utf-8 -*-
"""
step02_softmax_classifier_DNN.py

DNN model 
 - hidden layer : relu 활성함수 
 - output layer : Softmax 활성함수 
 - 2개 은닉층을 갖는 분류기 
 - hidden1_nodes = 12개 
 - hidden2_nodes = 6개 
 - dataset : iris
"""

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.x 사용 안함 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder, minmax_scale # y data 
from sklearn.metrics import accuracy_score # model 평가

# 1. x, y 공급 data 
iris = load_iris()

# x변수 : 1~4칼럼 
x_data = iris.data 
x_data.shape # (150, 4)

x_data = minmax_scale(x_data)

# y변수 : 5컬럼 
y_data = iris.target 
y_data.shape # (150,)

# reshape 
y_data = y_data.reshape(-1, 1)
y_data.shape # (150, 1)

'''
0 -> 1 0 0
1 -> 0 1 0
2 -> 0 0 1
'''

obj = OneHotEncoder()
# sparse -> numpy
y_data = obj.fit_transform(y_data).toarray()
y_data.shape # (150, 3)

# 75 vs 25
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data) 


# 2. X,Y변수 정의 
X = tf.placeholder(dtype=tf.float32, shape =[None,4]) # [관측치,입력수]
Y = tf.placeholder(dtype=tf.float32, shape =[None,3]) # [관측치,출력수]

#############################
## DNN network
#############################

hidden1_nodes = 12 
hidden2_nodes = 6

# hidden layer1 : 1층 : relu()
w1 = tf.Variable(tf.random_normal([4, hidden1_nodes]))#[input, output]
b1 = tf.Variable(tf.random_normal([hidden1_nodes])) # [output]
hidden1_output = tf.nn.relu(tf.matmul(X, w1) + b1)

# hidden layer2 : 2층 : relu()
w2 = tf.Variable(tf.random_normal([hidden1_nodes, hidden2_nodes]))#[input, output]
b2 = tf.Variable(tf.random_normal([hidden2_nodes])) # [output]
hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, w2) + b2)

# output layer : 3층 : sotfmax()
w3 = tf.Variable(tf.random_normal([hidden2_nodes, 3]))#[input, output]
b3 = tf.Variable(tf.random_normal([3])) # [output]

# 4. softmax 분류기 
# 1) 회귀방정식 : 예측치 
model = tf.matmul(hidden2_output, w3) + b3 # 회귀모델 -> 활성함수(relu) 

# softmax(예측치)
softmax = tf.nn.softmax(model) # 활성함수 적용(0~1) : y1:0.8,y2:0.1,y3:0.1

# (2) loss function : 

# 1차 방법 : Cross Entropy 이용 : -sum(Y * log(model))  
#loss = -tf.reduce_mean(Y * tf.log(softmax) + (1 - Y) * tf.log(1 - softmax))

# 2차 방법 : Softmmax + CrossEntropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = Y, logits = model))

# 3) optimizer : 오차 최소화(w, b update) 
train = tf.train.AdamOptimizer(0.1).minimize(loss) # 오차 최소화

# 4) argmax() : encoding(2) -> decoding(10)
y_pred = tf.argmax(softmax, axis = 1) # y1:0.8,y2:0.1,y3:0.1
y_true = tf.argmax(Y, axis = 1)


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
    y_pred_re = sess.run(y_pred, feed_dict = {X : x_data}) #예측치 
    y_true_re = sess.run(y_true, feed_dict = {Y : y_data}) # 정답

    #print("y pred =", y_pred_re)
    #print("y ture =", y_ture_re) 
       
    acc = accuracy_score(y_true_re, y_pred_re)
    print("accuracy =", acc) # accuracy = 0.98

