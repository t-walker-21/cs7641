import cv2
from cleanData import Dataset1
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np


data = Dataset1()

x,y = data.fetch_data_multi()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6000)


print len(X_train)
print len(y_train)


cls = tree.DecisionTreeClassifier(max_depth=2)

cls.fit(X_train,y_train)
tree.export_graphviz(cls,out_file='tree.dot')

inf = cls.predict(X_test)
gt = y_test.ravel()

correct = 0

for t in range(len(gt)):
    if inf[t] == gt[t]:
        correct += 1

print correct / (len(gt) * 1.0)