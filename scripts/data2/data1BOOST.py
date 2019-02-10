from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from gatherData import Dataset2
import sys

data = Dataset2()

dataSize = int(sys.argv[1])
x,y = data.fetch_data(dataSize)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=20)


print len(X_train)
print len(y_train)

cls = AdaBoostClassifier(tree.DecisionTreeClassifier(),n_estimators=10)

cls.fit(X_train,y_train.ravel())


inf = cls.predict(X_test)
gt = y_test.ravel()

correct = 0

for t in range(len(gt)):
    if inf[t] == gt[t]:
        correct += 1

print correct / (len(gt) * 1.0)