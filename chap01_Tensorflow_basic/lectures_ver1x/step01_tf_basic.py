# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:12:41 2020

@author: user
"""
#python 직접 실행 환경
x = 10
y = 20
z = x+y
print(z)

# import tensorflow as tf #ver 2.0
import tensorflow.compat.v1 as tf #ver 1.x
tf.disable_v2_behavior() #ver 2.x 사용
print(tf.__version__)


#tensorflow 간접 실행 환경 
'''프로그램 정의 영역 '''
x= tf.constant(10) #상수 정의
y= tf.constant(20) #상수 정의
print(x,y)

#Tensor("Const_4:0", shape=(), dtype=int32) 
#Tensor("Const_5:0", shape=(), dtype=int32)
#z= Tensor("add_2:0", shape=(), dtype=int32)

#식정의''' 
z= x+y
print("z=",z)
#z= Tensor("add_1:0", shape=(), dtype=int32)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
#session''' 
sess = tf.Session() #프로그램에서 정의한 상수, 변수, 식 ->device할당 (CPU, GPU, TPU)

'''프로그램 실행 영역 '''
print("x=",sess.run(x)) #x= 10
print("y=",sess.run(y)) #y= 20

#sess.run(x,y) #오류 발생 
x_val, y_val = sess.run([x,y])
print(x_val, y_val) #10 20

print("z=",sess.run(z)) # 상수 -> 연산 : z= 30

#객체 닫기 
sess.close()





