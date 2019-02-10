from gatherData import Dataset2
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
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
for depth in range (1,50):
        cls = tree.DecisionTreeClassifier(max_depth=depth)
        cls.fit(X_train,y_train)
        print "\nDepth: " , depth

        testPred = cls.predict(X_test)
        gt = y_test.ravel()
        correct = 0
        for t in range(len(gt)):
                if testPred[t] == gt[t]:
                        correct += 1

        accTest = correct / (len(gt) * 1.0)
        if accTest > maxTstAcc:
                maxTstAcc = accTest
                maxTstIdx = depth-1

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
                maxTrnIdx = depth-1
        print "Train performance  ",accTrain
        #print np.mean(cross_val_score(cls,x,y,cv=cv))
        depth_perf_train.append(accTrain)
        depth_perf_test.append(accTest)

plt.plot(depth_perf_train,label='Train')
plt.plot(depth_perf_test,label='Test')
plt.title('Validation Set Accuracy vs Tree Depth')
plt.xlabel('Max Depth')
plt.legend(loc='upper right')
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