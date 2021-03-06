'''
문) 다음 a와 b변수를 대상으로 행렬 브로드캐스팅 연산을 수행하여 출력하시오.
    조건1> a변수 : placeholder() 이용 
    조건2> b변수 : Variable()이용
    조건3> c변수 계산식 : c = a * b 
          -> tf.mat.multiply()이용 
    조건4> a,b,c변수 결과 출력    

<< 출력 결과 >> 
a= [ 1.  2.  3.] : 1x3
b= [[ 0.123]     : 3x1
    [ 0.234]
    [ 0.345]]
    
c= [[ 0.123       0.24600001  0.36900002]  : 3x3
   [ 0.234       0.46799999  0.70200002]
   [ 0.345       0.69        1.03499997]]
'''

import tensorflow as tf

import tensorflow.compat.v1 as tf #ver 1.x
tf.disable_v2_behavior() #ver 2.x 사용

a = [ 1.  2.  3.] 
print("a= ", a)
a =  tf.placeholder(dtype=tf.float32, shape=[1,3])


init = tf.global_variables_initializer()
b =  tf.Variable(dtype=tf.float32, shape=[3,1])
b = tf.constant([[0.123],[0.234],[0.345]])
print("b= ", b)

c = tf.multiply(a, b)
c =  tf.placeholder(dtype=tf.float32, shape=[3,3])
print("c= ", c)


