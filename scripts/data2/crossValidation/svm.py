from gatherData import Dataset2
from sklearn import svm
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

splits = 1
tst_size = 0.1

cls = svm.SVC(kernel='linear')
#cls = svm.SVC(kernel='rbf')
#cls = svm.SVC(kernel='poly',C=1)

cv = ShuffleSplit(n_splits=splits, test_size=tst_size,random_state=0)
#print cross_val_score(cls,x,y,cv=cv)
score = cross_val_score(cls,x,y,cv=cv)
print np.mean(score), np.std(score), "linear"

#cls = svm.SVC(kernel='linear')
#cls = svm.SVC(kernel='rbf')
cls = svm.SVC(kernel='poly')

cv = ShuffleSplit(n_splits=splits, test_size=tst_size,random_state=0)
#print cross_val_score(cls,x,y,cv=cv)
score = cross_val_score(cls,x,y,cv=cv)
print np.mean(score), np.std(score), "poly"


"""plt.plot(depth_perf)
plt.title('10-fold Cross-Validated Accuracy vs Tree Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.show()"""





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