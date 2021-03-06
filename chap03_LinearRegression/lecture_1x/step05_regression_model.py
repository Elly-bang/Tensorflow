# -*- coding: utf-8 -*-
"""
step05_regression_model.py : ver1.x
   - X(1) -> Y
   - 손실함수(loss function) : 오차 반환함수 
   - 모델 최적화 알고리즘 : 경사하강법알고리즘(GD, Adam) 적용 
     -> 모델 학습 : 최적의 기울기, 절편 -> loss값이 0에 수렴
"""

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.x 사용 안함 
import numpy as np
print(tf.__version__) # 2.0.0


# 1. X, Y data 생성 
#x_data = np.array([1,2,3]) # 입력 data 
#y_data = np.array([2,4,6]) # 출력 data 

# X변수 값이 큰 경우 : inf
x_data = np.array([1,2,3,125]) # 입력 data  -> 정규화 
y_data = np.array([2,4,6,250]) # 출력 data  -> 정규화 

# 0 ~ 1
x_data = x_data / 125 # [0.008, 0.016, 0.024, 1.   ]
# np.log
y_data = np.log(y_data) # [0.69314718, 1.38629436, 1.79175947, 5.52146092]

# 2. X, Y 변수 정의 
X = tf.placeholder(dtype=tf.float32, shape = [None]) # x_data 공급 
Y = tf.placeholder(dtype=tf.float32, shape = [None]) # y_data 공급 

# 3. a, b 변수 정의 : 최초 값 : 난수 시작 
a = tf.Variable(tf.random_normal([1])) # 기울기 
b = tf.Variable(tf.random_normal([1])) # 절편 

# 4. 식 정의 
model = tf.multiply(X, a) + b # 예측치 : 회귀방정식 
 
err = Y - model # 오차
 
loss = tf.reduce_mean(tf.square(err)) # 손실 함수

# 5. 최적화 객체 : 기울기, 절편 수정 
optimizer = tf.train.GradientDescentOptimizer(0.1) # 학습률 : 1~0

train = optimizer.minimize(loss) # 손실 최소화 : 최적의 기울기,절편 수정 

init = tf.global_variables_initializer()

# 6. 반복학습 
with tf.Session() as sess :
    sess.run(init) # 변수 초기화 : a, b 
    a_val, b_val = sess.run([a, b])
    print("최초 기울기와 절편 ")
    print("a = {}, b = {}".format(a_val, b_val))
    
    feed_data = {X : x_data, Y : y_data}
    
    # 반복학습 : 50회 
    for step in range(50) : 
        _, loss_val = sess.run([train, loss], feed_dict = feed_data)
        print("step =", (step+1), ", loss =", loss_val)
        a_val_up, b_val_up = sess.run([a, b])
        print("수정된 a = {}, b = {}".format(a_val_up, b_val_up))
        
    # model test 
    y_pred = sess.run(model, feed_dict = {X : [0.024]})  # X = 2.5  
    print("y pred =", y_pred) # y pred = [5.025069] -> y pred = [1.5129367]

'''
최초 기울기와 절편 
a = [-0.01355203], b = [-0.21065438]
step = 1 , loss = 20.661524
     :
step = 50 , loss = 0.0042284643
수정된 a = [2.1665504], b = [-0.3786083]
'''







