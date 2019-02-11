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
import sys
SIZE = 3*3*512

def get_highest(l):
	high = max(l)
	
	for n in range(len(l)):
		if l[n] == high:
			l[n] = 1
		else:
			l[n] = 0

	return l

	


INIT_LR = 1E-4
EPOCHS = 30
BS = 600
seed = 9068

np.random.seed(seed)
data = Dataset2()

dataSize = int(sys.argv[1])
x,y = data.fetch_data(dataSize)
del data



size_perf_train = []
size_perf_test = []

maxTrnAcc = 0
maxTrnIdx = 0
maxTstAcc = 0
maxTstIdx = 0

for data_size in range(1,10):


	size = 1 - (data_size * 0.1)
	print "test size: " , size
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=541211)

	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	print len(X_train) , " training samples"
	cls = Sequential()

	cls.add(Dense(100,input_dim=SIZE,activation='relu',kernel_initializer='random_uniform'))
	cls.add(Dense(len(y_train[0]),activation='softmax',kernel_initializer='random_uniform'))

	opt = Adam(lr=INIT_LR)

	cls.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

	cls.fit(X_train,y_train,epochs=EPOCHS,steps_per_epoch=10,verbose=0)

	inferenceTest = cls.predict(X_test)
	gt = y_test

	correct = 0

	for t in range(len(gt)):
		if (get_highest(inferenceTest[t]) == gt[t]).all():
			correct += 1

	#print correct
	score = correct / (len(gt) * 1.0)

	accTest = score
	if accTest > maxTstAcc:
			maxTstAcc = accTest
			maxTstIdx = data_size-1

	print "test performance  ",accTest


	inferenceTrain = cls.predict(X_train)
	gt = y_train

	correct = 0

	for t in range(len(gt)):
		if (get_highest(inferenceTrain[t]) == gt[t]).all():
			correct += 1

	#print correct
	score = correct / (len(gt) * 1.0)

	accTrain = score
	if accTrain > maxTrnAcc:
			maxTstAcc = accTest
			maxTstIdx = data_size-1

	accTrain = correct / (len(gt) * 1.0)
	if accTrain > maxTrnAcc:
			maxTrnAcc = accTrain
			maxTrnIdx = data_size-1
	print "Train performance  ",accTrain
	#print np.mean(cross_val_score(cls,x,y,cv=cv))
	size_perf_train.append(accTrain)
	size_perf_test.append(accTest)

plt.plot(size_perf_train,label='Train')
plt.plot(size_perf_test,label='Test')

plt.plot([maxTrnIdx,maxTstIdx],[maxTrnAcc,maxTstAcc],'r+')
plt.legend(loc='upper right')
plt.title('Accuracy vs Data Train Size')
plt.xlabel('Data Size (10s of percent)')
plt.ylabel('Accuracy')
#plt.xlim(1,9)
plt.show()

