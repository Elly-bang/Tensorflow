# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:47:36 2020

2. 초기값이 없는 변수. : feed방식
변수 = tf.placeholder(dtype, shape)
 - dtype : 자료형 (tf.float32, tf.int32, tf.string)
 - shape : 자료구조 ([n] : 1차원 , [r,c] : 2차원 , 생략 : 공급data결정)

"""

import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 



#변수 정의 
a = tf.placeholder(dtype=tf.float32) #shape생략 :가변형
b = tf.placeholder(dtype=tf.float32) #shape생략 :가변형

c = tf.placeholder(dtype=tf.float32, shape =[5]) #고정형 1d
d = tf.placeholder(dtype=tf.float32, shape =[None, 3]) #고정형 2d (행 : 가변형)

c_data = tf.random_uniform([5]) #0~1난수 

# 식 정의 
mul =tf.multiply(a, b)
add = tf.add(mul, 10)
c_calc = c *0.5 # vector * scala 

with tf.Session() as sess:
    #변수 초기화 생략 : variable형식 없음 모두 placeholder 임
    
    
    #식 실행 
    mul_re= sess.run(mul, feed_dict  = {a : 2.5, b: 3.5 })
    print("mul=", mul_re) #mul= 8.75
    
    #공급 data
    a_data = [1.0, 2.0, 3.5]
    b_data = [0.5, 0.3, 0.4]
    feed_data = {a : a_data, b : b_data}    
    
    mul_re2 = sess.run(mul, feed_dict= feed_data)
    print("mul_re2=", mul_re2) #[0.5 0.6 1.4]
    
    #식 실행 
    add_re = sess.run(add,feed_dict = feed_data) #mul+10 
    print("add=", add_re)
    
    # c_calc = c *0.5 # vector * scala 
    # c_data = tf.random_uniform([5]) #0~1난수
    c_data_re =  sess.run(c_data) #상수 생성 
    print( "c_calc=",sess.run(c_calc, feed_dict={c:c_data_re}))
    
    #c_calc= [0.02572292 0.23708719 0.3580258  0.17430085 0.48146087]
    
    '''
    주의 : 프로그램 정의 변수와 리턴 변수명은 다르ㄱㅔ 지정함
    '''
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    