# -*- coding: utf-8 -*-
"""
step03_MNIST_DNN.py

DNN model + NMIST + Hyper parameters + Mini batch
  - Network layer 
    input nodes : 28x28 = 784
    hidden1 nodes : 128 - 1층 
    hidden2 nodes : 64 - 2층 
    output nodes : 10 - 3층 
    
 - Hyper parameters  
   lr : 학습율 
   epochs : 전체 dataset 재사용 회수 
   batch size : 1회 data 공급 회수(mini batch) 
   iter size : 반복회수 
    -> 1epoch(60,000) : batch size(200) * iter size(200)
"""

import tensorflow.compat.v1 as tf # ver1.x
tf.disable_v2_behavior() # ver2.x 사용 안함 
from sklearn.preprocessing import OneHotEncoder # y data 
from sklearn.metrics import accuracy_score # model 평가
import matplotlib.pyplot as plt 
import numpy as np

# 1. MNIST dataset load 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train.shape # images(픽셀) : (60000, 28, 28) -(size, h, w)
y_train.shape # labels(10진수) # (60000,)

# 첫번째 image 확인 
plt.imshow(x_train[0]) # 5
plt.show()

x_train[0]
y_train[0] # 5

# 2. images 전처리 
# 1) images 정규화 
x_train_nor, x_test_nor = x_train / 255.0, x_test / 255.0

x_train_nor[0]
x_test_nor[0]

# 2) 3차원 -> 2차원 
x_train_nor = x_train_nor.reshape(-1, 784)
x_test_nor = x_test_nor.reshape(-1, 784)

x_train_nor.shape # (60000, 784)
x_test_nor.shape # (10000, 784)

# 3. labels 전처리
# 1) 1차원 -> 2차원  
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 2) ont-hot encoding 
obj = OneHotEncoder()
y_train_one = obj.fit_transform(y_train).toarray()
y_test_one = obj.fit_transform(y_test).toarray()
y_train_one.shape # (60000, 10)
y_test_one.shape # (10000, 10)


# 4. 변수 정의 
X = tf.placeholder(dtype = tf.float32, shape = [None, 784]) # x data
Y = tf.placeholder(dtype = tf.float32, shape = [None, 10]) # y data

#Hyper parameters  
lr = 0.01 #학습율 
epochs = 20 # 전체 dataset 재사용 회수 
batch_size = 200 # 1회 data 공급 회수(mini batch) 
iter_size = 300 # 반복회수 

####################################
## DNN network
####################################

hidden1_nodes = 128  
hidden2_nodes = 64 
 
# hidden layer1 : 1층 : relu()
w1 = tf.Variable(tf.random_normal([784, hidden1_nodes]))#[input, output]
b1 = tf.Variable(tf.random_normal([hidden1_nodes])) # [output]
hidden1_output = tf.nn.relu(tf.matmul(X, w1) + b1)

# hidden layer2 : 2층 : relu()
w2 = tf.Variable(tf.random_normal([hidden1_nodes, hidden2_nodes]))#[input, output]
b2 = tf.Variable(tf.random_normal([hidden2_nodes])) # [output]
hidden2_output = tf.nn.relu(tf.matmul(hidden1_output, w2) + b2)

# output layer : 3층 : sotfmax()
w3 = tf.Variable(tf.random_normal([hidden2_nodes, 10]))#[input, output]
b3 = tf.Variable(tf.random_normal([10])) # [output]

   
# 5. softmax 알고리즘 
# (1) model
model = tf.matmul(hidden2_output, w3) + b3

# (2) softmax
softmax = tf.nn.softmax(model) # 활성함수 

# (3) loss function : Softmaxt + Cross Entorpy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = Y, logits = model))

# (4) optimizer 
train = tf.train.AdamOptimizer(lr).minimize(loss) 

# (5) encoding -> decoding 
y_pred = tf.argmax(softmax, axis = 1)
y_true = tf.argmax(Y, axis = 1)


# 6. model training
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # w, b 초기화 
    
    feed_data = {X : x_train_nor, Y : y_train_one}
    
    # epochs = 20 
    for epoch in range(epochs) : # 1세대 
        tot_loss = 0
        
        # 1epoch = 200 * 300
        for step in range(iter_size) :  # 300반복 학습 
            idx = np.random.choice(a=y_train_one.shape[0], 
                                   size=batch_size, replace=False)
            # Mini batch dataset 
            feed_data = {X : x_train_nor[idx], Y : y_train_one[idx]}
            _, loss_val = sess.run([train, loss], feed_dict = feed_data)
            
            tot_loss += loss_val
            
        # 1epoch 종료 
        avg_loss = tot_loss / iter_size
        print("epoch = {}, loss = {}".format(epoch+1, avg_loss))         
        
    # model 최적화 : test 
    feed_data2 = {X : x_test_nor, Y : y_test_one}
    y_pred_re = sess.run(y_pred, feed_dict = feed_data2)
    y_true_re = sess.run(y_true, feed_dict = feed_data2)
    
    acc = accuracy_score(y_true_re, y_pred_re)
    print("accuracy =", acc)
            
    
    









