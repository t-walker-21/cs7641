from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from cleanData import Dataset1
import numpy as np
import time

data = Dataset1()

x,y = data.fetch_data_multi()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=412)


print len(X_train)
print len(y_train)

size_perf = []
for data_size in np.arange(1,10,1):
        cls = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),n_estimators=50)
        size = 1 - (data_size) * 0.1
        print "size: " , size
        cv = ShuffleSplit(n_splits=1, test_size=size,random_state=102)
        score = cross_val_score(cls,x,y,cv=cv)
        #print np.mean(cross_val_score(cls,x,y,cv=cv))
        size_perf.append(np.mean(score))

plt.plot(size_perf)
plt.title('Accuracy vs Data Train Size')
plt.xlabel('Data Size (percent)')
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