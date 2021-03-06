from cleanData import Dataset1
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
import numpy as np
from matplotlib import pyplot as plt

data = Dataset1()

x,y = data.fetch_data_multi()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


print len(X_train)
print len(y_train)


depth_perf = []
for depth in range (1,60):
        cls = tree.DecisionTreeClassifier(max_depth=depth)

        cv = ShuffleSplit(n_splits=10, test_size=0.2,random_state=0)
        score = cross_val_score(cls,x,y,cv=cv)
        #print np.mean(cross_val_score(cls,x,y,cv=cv))
        depth_perf.append(np.mean(score))

plt.plot(depth_perf)
plt.title('10-fold Cross-Validated Accuracy vs Tree Depth')
plt.xlabel('Max Depth')
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