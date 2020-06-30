# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:01:47 2020

name_scope : 영역별 tensorflow시각화
-model생성 -> 모델 오차 -> 모델 평가

"""
import tensorflow.compat.v1 as tf #ver 1.x
tf.disable_v2_behavior() #ver 2.x 사용

#초기화
tf.reset_default_graph()

#상수 정의 :X, a, b, Y
X = tf.constant(5.0, name="x_data") #입력값
a = tf.constant(10.0, name ="a") #기울기
b = tf.constant(4.45, name="b") #절편
Y = tf.constant(55.0, name="Y") #결과값

#회귀방정식 name_scope
with tf.name_scope("Regress_model") as scope:
     model = (X*a)+b #y예측치 
    
with tf.name_scope("model_error") as scope:
     model_err =tf.abs(tf.subtract(Y, model)) #부호 절대값

with tf.name_scope("Model_evaluation") as scope :
     square = tf.square(model_err)
     mse =tf.reduce_mean( tf.square (tf.subtract(Y, model ))) #mse
    
with tf.Session() as sess:
     tf.summary.merge_all() #tensor모으는 역할 
     writer = tf.summary.FileWriter("C:/ITWILL/6_Tensorflow/graph", sess.graph)
     writer.close()
     print("X=", sess.run(X)) #X= 5.0
     print("Y=", sess.run(Y)) #Y= 55.0
     print("y pred=", sess.run(model)) #54.45
     print("model_err=", sess.run(model_err)) # 0.54999924
     print("mse=", sess.run(mse)) #0.30249918