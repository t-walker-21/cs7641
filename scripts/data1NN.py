import cv2
import numpy as np
import os
from keras import applications
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from sklearn.cross_validation import train_test_split
from sklearn import svm
from cleanData import Dataset1
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

INIT_LR = 1E-3
EPOCHS = 500
BS = 10

data = Dataset1()

x,y = data.fetch_data()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=90)
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)

print len(X_train)
print len(y_train)


cls = Sequential()

cls.add(Dense(7,input_dim=7,activation='sigmoid'))
#cls.add(Dense(10,activation='sigmoid'))

cls.add(Dense(1,activation='sigmoid'))

opt = Adam(lr=INIT_LR)

cls.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])





print ("TRAINING MODEL")

cls.fit(X_train,y_train,epochs=EPOCHS,batch_size=BS)


inf = cls.predict(X_test)
gt = y_train.ravel()
print inf

correct = 0
"""
for t in range(len(gt)):
    if inf[t] == gt[t]:
        correct += 1

print correct
print "score", correct / (len(gt) * 1.0)"""