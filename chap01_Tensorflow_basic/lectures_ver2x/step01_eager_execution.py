# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:55:23 2020

step01_eager_execution.py

tensorflow 2.0 
1. 즉시 실행(eager_execution) 환경
 - session object없이 즉시 실행 환경(auto graph)
 - python실행 환경과 동일
 - API 정리 : tf.global_variables_initializer() 삭제
"""
import tensorflow as tf # ver 2.0 
print(tf.__version__) #2.0.0

#상수 정의 : session없이 작동
a = tf.constant([[1,2,3],[1.0,2.5,3.5]]) #[2,3]
print("a:",a)

'''
tf.Tensor(
[[1.  2.  3. ]
 [1.  2.5 3.5]], shape=(2, 3), dtype=float32)
'''

print(a.numpy()) #데이터만 추출하고 싶을때

'''
[[1.  2.  3. ]
 [1.  2.5 3.5]]
'''
                                                                                                                                                                                                                                                                                                                                                      
#식 정의  : 상수 참조 -> 즉시 실행
b= tf.add(a,0.5)
print("b:")
print(b)

#변수 정의 
x = tf.Variable([10,20,30])
y = tf.Variable([1,2,3])
mul=tf.multiply(x,y)

print(x.numpy()) #[10 20 30]
print(y.numpy()) #[1 2 3]
print(mul) #tf.Tensor([10 40 90], shape=(3,), dtype=int32)
print(mul.numpy()) #[10 40 90]


# python code -> tensorflow 실행 
x = [[2.0, 3.0]]
a = [[1.0], [1.5]]


#행렬곱 실행
mat = tf.matmul(x, a)
print("matrix multiply = {} ". format(mat)) #matrix multiply = [[6.5]] 



'''step02_tf_init -> ver 2.0'''

print("##auto graph##")
print("##tf_ver1.0 -> ver 2.0## ")
'''프로그램 정의 영역'''
#상수 정의 
x  = tf.constant([1.5, 2.5, 3.5],name='x') #1차원의 자료를 저장하는 수정 불가능한 상수 = x

#변수 정의 
y = tf.Variable([1.0, 2.0, 3.0], name='y') #1차원 수정 가능 상수 y 

#식 정의
mul = x * y #상수 * 변수  

'''프로그램 실행 영역'''

print("x=",x.numpy()) #상수 할당 : x= [1.5 2.5 3.5]
print("y=",y.numpy()) #변수 할당 : y= [1. 2. 3.] 

#식 할당
print("mul=",mul.numpy()) #mul= [ 1.5  5.  10.5]




















