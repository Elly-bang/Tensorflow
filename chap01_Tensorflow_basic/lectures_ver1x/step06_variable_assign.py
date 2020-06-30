# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:37:16 2020
step06_variable_assign.py

난수 상수 생성 함수 : 정규분포난수, 균등분포 난수
tf.Variable(난수 상수) -> 변수 값 수정
"""


import tensorflow.compat.v1 as tf # ver 1.x
tf.disable_v2_behavior() # ver 2.x 사용안함 


# 난수
num = tf.constant(10.0)

# 1차원(scala) 변수
var = tf.Variable(num + 20.0)  # 상수 + 상수 = scala
print("var =", var)  # var = <tf.Variable 'Variable_10:0' shape=() dtype=float32_ref>

# 1차원 변수
var1d = tf.Variable(tf.random_normal([3]))  # 1차원 : [n]
print("var1d =", var1d)  # var1d = <tf.Variable 'Variable_12:0' shape=(3,) dtype=float32_ref>

# 2차원 변수
var2d = tf.Variable(tf.random_uniform([3,2]))  # 2차원 : [r,c]
print("var2d =", var2d)

# 3차원 변수
var3d = tf.Variable(tf.random_normal([3,2,4]))  # 3차원 : [s,r,c]
print("var3= ", var3d)  # var3=  <tf.Variable 'Variable_28:0' shape=(3, 2, 4) dtype=float32_ref>


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  # 변수 초기화(초기값을 할당) : var, var1, var2
    
    # 얘네는 난수라 실행시마다 값이 바뀜
    print("var =", sess.run(var))  # var = 30.0
    print("var1 =", sess.run(var1d))  # var1 = [ 0.6282741  1.1691085 -0.3373233]
    print("var2 =", sess.run(var2d))
    '''
    var2 = [[-0.7900748  -1.4926518 ]
            [ 0.20589611  0.17302911]
            [-0.5435695   2.0427992 ]]
    '''
    
    # 변수 값 수정
    var1d_data = [0.1, 0.2, 0.3]
    print("var1d assign_add", sess.run(var1d.assign_add(var1d_data)))  # 벡터에는 벡터를 더한다.
    # var1d assign_add [0.49912205 0.60455304 0.02951023]
    print("var1d assign =", sess.run(var1d.assign(var1d_data)))  # var1d assign = [0.1 0.2 0.3]
    print("var3d =", sess.run(var3d))
    '''
    var3d = [[[ 2.3649235   0.40684322 -0.76584905 -0.73394614]
              [ 0.5106442   0.50091386  0.16552165 -1.2709608 ]]
            
             [[ 0.72862685 -0.26156664 -0.04909089  0.04150959]
              [ 1.0194414   0.36773545 -0.31673393 -0.63394797]]
            
             [[-0.5678536   1.3605262   0.35045168 -0.78765357]
              [ 0.5804654  -1.5777429   0.26768118 -2.1375074 ]]]
    '''
    
    var3d_re = sess.run(var3d)
    print(var3d_re[0].sum())  # 첫번째 면 : 합계  # -1.0675739
    print(var3d_re[0,0].mean())  # 첫 번째 면, 첫 번재 행 : 평균  # -0.39534742
    
    # 24개 균등분포난수를 생성하여 var3d변수에 값을 수정하시오.
    var3d_data = tf.random_uniform([3,2,4])
    print("var3d assign =", sess.run(var3d.assign(var3d_data)))
    '''
    var3d assign = [[[0.2649839  0.24260855 0.50324714 0.70864356]
                      [0.206092   0.86252224 0.7500526  0.57766306]]
                    
                     [[0.8154572  0.6812402  0.36573362 0.36797452]
                      [0.6343192  0.58422804 0.8183737  0.39555907]]
                    
                     [[0.3842764  0.95449173 0.77604914 0.5163716 ]
                      [0.23464894 0.9567236  0.77383673 0.3961681 ]]]
    '''
    
    
    
































