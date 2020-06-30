# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:00:22 2020

step04_@tf.function2.py

- Tensorflow 2.0 
- 3_@tf.function 함수 장식자 (데코레이터) 
         여러 함수를 포함하는  main함수 
"""
import tensorflow as tf

# model생성 함수 1
def linear_model (x) :
    return x * 2 + 0.2 #회귀

# model 오차 함수 2
def model_err(y, y_pred):
    return y - y_pred


#model 평가 함수 
@tf.function    
def model_evaluation(x,y):
    y_pred = linear_model(x) #1호출
    err = model_err(y, y_pred ) #2호출  
    return tf.reduce_mean(tf.square(err)) #mse

#x, y data생성 
 X = tf.constant([1,2,3], dtype= tf.float32)   
 Y = tf.constant([2,4,6], dtype= tf.float32)
 MSE = model_evaluation(X, Y)
 print("MSE=%.5f"%( MSE)) #MSE=0.04000 



