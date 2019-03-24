import numpy as np
import os
from keras import applications
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras.utils import to_categorical
	


INIT_LR = 5E-4
EPOCHS = 500
BS = 50
seed = 9068

np.random.seed(seed)

fin = open("../data1.txt","r")

X = []
y = []

for l in fin:
    X.append(l.split(",")[:-2])
    y.append(int(l.split(",")[-2]))

X = np.array(X,dtype=np.float32)

y = to_categorical(y)

cls = Sequential()

cls.add(Dense(7,input_dim=7,activation='relu',kernel_initializer='random_uniform'))
cls.add(Dense(30,activation='relu',kernel_initializer='random_uniform'))
cls.add(Dense(3,activation='softmax',kernel_initializer='random_uniform'))

opt = Adam(lr=INIT_LR)

cls.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

history = cls.fit(X,y,epochs=EPOCHS,steps_per_epoch=10,validation_split=0.2,validation_steps=50)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()