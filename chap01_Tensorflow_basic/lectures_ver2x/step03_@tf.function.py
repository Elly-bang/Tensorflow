# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:36:59 2020

step03_@tf.function.py

- Tensorflow 2.0 
- 3_@tf.function 함수 장식자 (데코레이터) 
 - 함수 장식자 이점 : (=inner함수와 비슷)
     -> python code -> tensorflow code변환 
     -> logic 처리 : 쉬운 코드 대체 
     -> 속도 향상 
"""
import tensorflow as tf 
import numpy as np
'''step09_tf_logic.py -> ver2.0 

# 1. if문 
x = tf.constant(10)

def true_fn() :
    return tf.multiply(x, 10) #x*10

def false_fn():
    return tf.add(x, 10) #x+10

#y = tf.cond(pred, true_fn, false_fn)
y = tf.cond( x > 100 , true_fn, false_fn) #false 


# while
i = tf.constant(0) #i=0 :반복변수

def cond(i) :
    return tf.less(i, 100) # i < 100  100 전까지 반복수행 

def body(i):
    return tf.add(i,1) # cond만족하는 것이 끝나면 i = i+1 까지 반복수행

 

loop = tf.while_loop(cond, body, (i,)) # tuple (i,) list [i]


sess = tf.Session()

print("y=", sess.run(y)) #y= 20
print("loop=", sess.run(loop)) #loop= 100
'''

@tf.function
def if_func(x):
    # python -> tensorflow code
    if x > 100 :
       y= x * 10 
    else :
       y= x + 10
    return y

x = tf.constant(10) 
        
# if_func(x) 으로 값 알아보기        
print("y= ", if_func(x).numpy()) #y=  20

@tf.function
def while_func(i):
    while i <100:
        i +=1 #i=i+1
    return i

i = tf.constant(0)
print("loop=",while_func(i).numpy()) #loop= 100












