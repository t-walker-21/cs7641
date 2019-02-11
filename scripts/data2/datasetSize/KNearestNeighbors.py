from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from gatherData import Dataset2
import numpy as np
import time
import sys

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
        cls = KNeighborsClassifier(n_neighbors=1,metric='manhattan')
        size = 1 - (data_size * 0.1)
        print "test size: " , size
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=52)
        cls.fit(X_train,y_train)

        testPred = cls.predict(X_test)
        gt = y_test.ravel()

        correct = 0

        for t in range(len(gt)):
                if testPred[t] == gt[t]:
                        correct += 1

        accTest = correct / (len(gt) * 1.0)
        if accTest > maxTstAcc:
                maxTstAcc = accTest
                maxTstIdx = data_size-1

        print "test performance  ",accTest


        trainPred = cls.predict(X_train)
        gt = y_train.ravel()

        correct = 0

        for t in range(len(gt)):
                if trainPred[t] == gt[t]:
                        correct += 1

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
plt.show()

"""

cls.fit(X_train,y_train.ravel())


inf = cls.predict(X_test)
gt = y_test.ravel()

correct = 0

for t in range(len(gt)):
    if inf[t] == gt[t]:
        correct += 1

print correct / (len(gt) * 1.0)

#mat = confusion_matrix(cls.predict(X_test),y_test)


#print(mat)

plt.matshow(mat)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()"""