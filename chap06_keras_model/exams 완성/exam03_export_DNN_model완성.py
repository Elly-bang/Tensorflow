# -*- coding: utf-8 -*-
"""
문) cifar10 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하기 위해서
    단계2. Dataset 생성과 단계3. Model 클래스 내용을 채우시오.
    
  조건1> keras layer
       L1 =  (32, 32, 3) x 256
       L2 =  256 x 128
       L3 =  128 x 64
       L4 =  64 x 10
  조건2> output layer 활성함수 : softmax     
  조건3> optimizer = 'Adam',
  조건4> loss = 'categorical_crossentropy'
  조건5> metrics = 'accuracy'
  조건6> epochs = 20, batch_size = 50 
  
<츨력 예시>  
Epoch 1, Train Loss: 2.19101, Train Acc: 0.25788, Test Loss: 2.13325, Test Acc: 0.31580
Epoch 2, Train Loss: 2.12259, Train Acc: 0.33052, Test Loss: 2.12327, Test Acc: 0.33180
Epoch 3, Train Loss: 2.10337, Train Acc: 0.35070, Test Loss: 2.10720, Test Acc: 0.34430
Epoch 4, Train Loss: 2.08615, Train Acc: 0.36664, Test Loss: 2.07554, Test Acc: 0.38020
Epoch 5, Train Loss: 2.08072, Train Acc: 0.37330, Test Loss: 2.07430, Test Acc: 0.37800
''' 생략 '''
"""


import tensorflow as tf
from tensorflow.python.data import Dataset # dataset 생성 
from tensorflow.keras.layers import Dense, Flatten # layer 구축 
from tensorflow.keras.datasets.cifar10 import load_data # Cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras import losses,optimizers,metrics#손실,최적화,평가 

# 단계1. dataset load & preprocessing
print('data loading')
(x_train, y_train), (x_val, y_val) = load_data()

print(x_train.shape) # (50000, 32, 32, 3) : 4차원 
print(y_train.shape) # (50000, 1) : 2차원 

# x_data 전처리 : 0~1 정규화
x_train = x_train / 255.0
x_val = x_val / 255.0



# 단계2. Dataset 생성 
train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(50)
train_ds # ((None, 28, 28), (None,)), types: (tf.uint8, tf.uint8)>
test_ds = Dataset.from_tensor_slices((x_val, y_val)).batch(50)
test_ds # ((None, 28, 28), (None,)), types: (tf.uint8, tf.uint8)>


input_shape = (32, 32, 3)

# 단계3. 순방향 step : model layer 정의  
class Model(tf.keras.Model): # tf.keras Model 상속 
    def __init__(self) : # 생성자 
        super().__init__() # 부모생성자 호출 
        # w, b 자동 초기화 
        # DNN layer
        # 3d(32, 32, 3) -> 1d(32 * 32 * 3)
        self.d0 = Flatten(input_shape = input_shape) # flatten : 2d image
        self.d1 = Dense(256, activation='relu') # hidden layer1
        self.d2 = Dense(128, activation='relu') # hidden layer2
        self.d3 = Dense(64, activation='relu') # hidden layer3
        self.d4 = Dense(10, activation='softmax') # output layer
     
    def call(self, inputs) : # inputs = images(32) : object(inputs)
        # 회귀방정식 생략 
        x = self.d0(inputs)
        x = self.d1(x) 
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x) # 예측치(확률) 반환 


# 단계4. loss function : losses 모듈 대체 
loss = losses.SparseCategoricalCrossentropy(from_logits=True)


# 단계5. model & optimizer
model = Model()
optimizer = optimizers.Adam() # lr : 자동설정(lr=0.1)


# 단계6. model test : 1epoch -> train/test loss and accuracy 측정 
train_loss = metrics.Mean() # 전체 원소 -> 평균 객체 반환 
train_acc = metrics.SparseCategoricalAccuracy() # 분류정확도 객체 반환
  
test_loss = metrics.Mean()
test_acc = metrics.SparseCategoricalAccuracy()


# 단계7. 역방향 step : tf.GradientTape 클래스 이용 
@tf.function # 연산 속도 향상 
def train_step(images, labels): # train step
    with tf.GradientTape() as tape:
        # 1) 순방향    
        preds = model(images, training=True) # 예측치  
        loss_value = loss(labels, preds) # 손실값 
        
        # 2) 역방향 
        grads = tape.gradient(loss_value, model.trainable_variables)
        # model optimizer
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # 3) 1epoch model training test : loss, acc save 
        train_loss(loss_value) # 1epoch loss 평균 : object(params) 
        train_acc(labels, preds) # 1epoch accuracy : object(params) 
  
@tf.function # # 연산 속도 향상
def test_step(images, labels): # test step
    preds = model(images, training=False) # 최적화된 model -> 예측치
    loss_value = loss(labels, preds) # 손실값 
    
    # 1epoch loss, acc save 
    test_loss(loss_value)
    test_acc(labels, preds)


epochs = 20

# 단계8. model training
for epoch in range(epochs):
  train_loss.reset_states()
  train_acc.reset_states()
  test_loss.reset_states()
  test_acc.reset_states()
  
  # train/test loss and acc per 1epoch
  for images, labels in train_ds: # training
    train_step(images, labels)

  for test_images, test_labels in test_ds: # testing
    test_step(test_images, test_labels)

  form = 'Epoch {}, Train Loss: {:.5f}, Train Accuracy: {:.5f}, Test Loss: {:.5f}, Test Accuracy: {:.5f}'
  print(form.format(epoch+1, train_loss.result(),
                        train_acc.result(),
                        test_loss.result(),
                        test_acc.result()))  
