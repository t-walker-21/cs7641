import numpy as np
import os
from keras import applications
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn import svm
from gatherData import Dataset2
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from matplotlib import pyplot as plt
SIZE = 7*7*512

def get_highest(l):
	high = max(l)
	
	for n in range(len(l)):
		if l[n] == high:
			l[n] = 1
		else:
			l[n] = 0

	return l

	


INIT_LR = 1E-5
EPOCHS = 25
BS = 50

data = Dataset2()

x,y = data.fetch_data()
print "in train func"
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=9)



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



print len(X_train)
print len(y_train)



cls = Sequential()

cls.add(Dense(100,input_dim=SIZE,activation='relu'))

cls.add(Dense(len(y_train[0]),activation='softmax',kernel_initializer='random_uniform'))

opt = Adam(lr=INIT_LR)

cls.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])





print ("TRAINING MODEL")

history = cls.fit(X_train,y_train,epochs=EPOCHS,steps_per_epoch=10,validation_split=0.2,validation_steps=10)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


inf = cls.predict(X_test)
gt = y_test

print inf

"""inf2 = []
for num in inf:
	if num >= 0.5:
		inf2.append(1)
	else:
		inf2.append(0)
inf2 = np.array(inf2)"""

#print inf2

correct = 0


for t in range(len(gt)):
    if (get_highest(inf[t]) == gt[t]).all():
        correct += 1

print correct
print "score", correct / (len(gt) * 1.0)