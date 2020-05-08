# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 07:51:51 2020

@author: Krishna
"""
#%%
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#%%
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape)

#%%
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)

y_train = keras.utils.to_categorical(y_train, num_classes = 10)
y_test = keras.utils.to_categorical(y_test,num_classes = 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape: ',x_train.shape)
print(x_train.shape[0], 'Train samples')
print(x_test.shape[0],'Test Samples')

#%%
from keras import metrics

#%%
batch_size = 128
num_classes = 10
epochs = 10

model = Sequential()
model.add(Conv2D(32,kernel_size = (3,3),activation = 'relu', 
                 input_shape = (28,28,1)))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation = 'softmax'))

l = keras.losses.categorical_crossentropy
from keras.optimizers import Adadelta
o =Adadelta(learning_rate = 1.0,rho = 0.95)

model.compile(loss = l,optimizer = o,metrics=['accuracy'])

#%%
hist = model.fit(x_train,y_train,batch_size= batch_size,epochs = epochs
                 , verbose=1,validation_data=(x_test,y_test))
print("The model has been trained successfully")

model.save('mnist.h5')
print("Model saved as mnist.h5")

#%%
score = model.evaluate(x_test,y_test,verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ',score[1])

