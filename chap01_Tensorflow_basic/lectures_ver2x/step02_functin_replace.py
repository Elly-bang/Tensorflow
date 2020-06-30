# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:57:44 2020

step02_functin_replace.py

- Tensorflow 2.0 

2. 세션 대신 함수 
- ver 2.0 : python 함수 사용 권장 
- API 정리 : tf.placeholder() 삭제 : 함수 인수 대체 
            ver1.x: tf.random_uniform-> ver2.x : tf.random.uniform()
            ver1.x: tf.random_normal -> ver2.x : tf.random.normal()
         
"""
import tensorflow as tf

'''step07_variable_feed.py'''

'''
#변수 정의 
a = tf.placeholder(dtype=tf.float32) #shape생략 :가변형
b = tf.placeholder(dtype=tf.float32) #shape생략 :가변형

c = tf.placeholder(dtype=tf.float32, shape =[5]) #고정형 1d
d = tf.placeholder(dtype=tf.float32, shape =[None, 3]) #고정형 2d (행 : 가변형)

c_data = tf.random_uniform([5]) #0~1난수 

# 식 정의 
mul = tf.multiply(a, b)
add = tf.add(mul, 10)
c_calc = c *0.5 # vector * scala 
'''

def mul_fn(a, b): # 인수 tf.placeholder() -> 인수 대체 
    return  tf.multiply(a, b)


def add_fn(mul) :
    return tf.add(mul,10)


def c_clac(c) :
    return c * 0.5                #tf.multiply(c,0.5)

#data 
a_data =[1.0, 2.5, 3.5]
b_data = [2.0, 3.0, 4.0]
mul_re = mul_fn(a_data,b_data)   
print("mul" , mul_re.numpy())
print("add={}".format(add_fn(mul_re)))


#tf.random.uniform() # ver1.x: random.normal -> ver2.x : tf.random.uniform() 
c_data =tf.random.uniform(shape = [3,4], minval=0, maxval=1) #ver2.x
print(c_data)

'''
[[0.7145798  0.59743476 0.33175588 0.57080865]
 [0.59893286 0.61455584 0.2841164  0.21442008]
 [0.5178684  0.8110846  0.06802213 0.14526129]], shape=(3, 4), dtype=float32)
    '''
 print("c_clac_function : ")
 print( c_clac(c_data).numpy())   
'''
[[0.02974403 0.38609892 0.0157612  0.25504076]
 [0.029805   0.22677624 0.44836313 0.08108479]
 [0.09695053 0.04057527 0.40015876 0.479865  ]]
'''



