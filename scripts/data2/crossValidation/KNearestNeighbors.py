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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=60)

print len(x)
print len(y)

maxTstAcc = 0
maxTstIdx = 0
maxTrnAcc = 0
maxTrnIdx = 0
depth_perf_train = []
depth_perf_test = []
for neigh in range (1,20):
        cls = KNeighborsClassifier(n_neighbors=neigh,metric='manhattan')
        cls.fit(X_train,y_train)
        print "\nNeighbors: " , neigh

        testPred = cls.predict(X_test)
        gt = y_test.ravel()
        correct = 0
        for t in range(len(gt)):
                if testPred[t] == gt[t]:
                        correct += 1

        accTest = correct / (len(gt) * 1.0)
        if accTest > maxTstAcc:
                maxTstAcc = accTest
                maxTstIdx = neigh-1

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
                maxTrnIdx = neigh-1
        print "Train performance  ",accTrain
        #print np.mean(cross_val_score(cls,x,y,cv=cv))
        depth_perf_train.append(accTrain)
        depth_perf_test.append(accTest)

plt.plot(depth_perf_train,label='Train')
plt.plot(depth_perf_test,label='Test')
plt.title('Validation Set Accuracy vs Number of Neighbors')
plt.xlabel('Number of neighbors')
plt.legend(loc='upper right')
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