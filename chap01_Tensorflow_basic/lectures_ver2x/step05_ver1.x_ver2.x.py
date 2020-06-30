# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:36:24 2020


1. 즉시 실행 모드
2. 세션 대신 함수 
3. @tf.function 함수 장식자 (데코레이터) 
"""
import tensorflow as tf

import pandas as pd #csv file load
from sklearn.model_selection import train_test_split  #data split

iris = pd.read_csv(("C:/ITWILL/6_Tensorflow/data/iris.csv"))
iris.info()

# 1. 공급data생성
cols = list(iris.columns)
x_data = iris [cols[:4]]
y_data = iris [cols[-1]]

x_data.shape #(150, 4)
y_data.shape #(150, )

# 2. X,Y 변수 정의 : tensorflow 변수 정의 

# 3. train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.3)

x_train.shape #(105, 4)
x_test.shape #(45, 4)



