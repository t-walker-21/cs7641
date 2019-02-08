import cv2
import numpy as np
import os
from keras import applications
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from sklearn.datasets import fetch_mldata
from keras.utils import to_categorical
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import svm
from numpy import argmax

INIT_LR = 1E-4
EPOCHS = 5
BS = 32

mnist = fetch_mldata('MNIST original')

x = []
y = []
 
for i in range(0,len(mnist.target)):

 x.append(mnist.data[i]) 
 y.append(mnist.target[i])


enc = to_categorical(y)

x = np.asarray(x)
y = np.asarray(enc)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


cls = Sequential()

cls.add(Dense(128,input_dim=784,activation='relu'))
cls.add(Dense(64,activation='relu'))

cls.add(Dense(10,activation='softmax'))

opt = Adam(lr=INIT_LR)

cls.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])





print ("TRAINING MODEL")

cls.fit(x,y,epochs=EPOCHS,batch_size=BS)



for t in range(0,len(y_test)):
 guess = []
 guess.append(X_test[t])
 guess = np.asarray(guess)

 cv2.imshow(str(argmax(cls.predict(guess)[0])),X_test[t].reshape(28,28))
 cv2.waitKey(700)
