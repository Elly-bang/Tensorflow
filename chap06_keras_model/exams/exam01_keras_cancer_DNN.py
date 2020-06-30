'''
문) breast_cancer 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
  조건1> keras layer
       L1 =  30 x 64
       L2 =  64 x 32
       L3 =  32 x 2
  조건2> optimizer = 'adam',
  조건3> loss = 'binary_crossentropy'
  조건4> metrics = 'accuracy'
  조건5> epochs = 300 
'''

import tensorflow as tf # ver2.x
from sklearn.datasets import load_breast_cancer # data set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, OneHotEncoder
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense # DNN layer 

# 1. breast_cancer data load
cancer = load_breast_cancer()

x_data = cancer.data
y_data = cancer.target
print(x_data.shape) # (569, 30) : matrix
print(y_data.shape) # (569,) : vector

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding 
obj = OneHotEncoder()
# reshape(2d) -> one hot encoding -> numpy
y_one_hot = obj.fit_transform(y_data.reshape([-1, 1])).toarray() 
y_one_hot
'''
[1, 00]
[0, 10]
'''
y_one_hot.shape # (569, 2)


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_one_hot, test_size = 0.3)


# 3. keras model


# 4. DNN model layer 구축 


# 5. model compile 


# 6. model trainint 


# 7. model evaluation 

