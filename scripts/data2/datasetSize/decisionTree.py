from gatherData import Dataset2
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
import numpy as np
from matplotlib import pyplot as plt
import sys

data = Dataset2()

dataSize = int(sys.argv[1])
x,y = data.fetch_data(dataSize)
del data


print len(x)
print len(y)
#exit()

size_perf_train = []
size_perf_test = []

maxTrnAcc = 0
maxTrnIdx = 0
maxTstAcc = 0
maxTstIdx = 0

for data_size in range(1,10):
        cls = tree.DecisionTreeClassifier(max_depth=79)
        size = 1 - (data_size * 0.1)
        print "test size: " , size
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=21)
        print len(X_train) , " training samples"
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

cls.fit(X_train,y_train)
tree.export_graphviz(cls,out_file='tree.dot')

inf = cls.predict(X_test)
gt = y_test.ravel()

correct = 0

for t in range(len(gt)):
        if inf[t] == gt[t]:
                correct += 1

acc = correct / (len(gt) * 1.0)
print "test performance  ",acc



depth_perf = []
for depth in range (1,60):
        cls = tree.DecisionTreeClassifier(max_depth=depth)

        cls.fit(X_train,y_train)
        tree.export_graphviz(cls,out_file='tree.dot')

        inf = cls.predict(X_train)
        gt = y_train.ravel()

        correct = 0

        for t in range(len(gt)):
                if inf[t] == gt[t]:
                        correct += 1
        acc = correct / (len(gt) * 1.0)
        print "test performance for depth: ", depth, ": " , acc
        depth_perf.append(acc)

plt.plot(depth_perf)
plt.show()"""