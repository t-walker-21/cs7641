from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
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


#cls = KNeighborsClassifier(n_neighbors=1,metric="manhattan")
cls = KNeighborsClassifier(n_neighbors=1,metric="manhattan")

neighbor_perf = []
neighbor_perf_man = []
for neigh in range (1,30):
        cls = KNeighborsClassifier(n_neighbors=neigh,metric="manhattan")
        cls2 = KNeighborsClassifier(n_neighbors=neigh)

        cv = ShuffleSplit(n_splits=10, test_size=0.2,random_state=0)
        score = np.mean(cross_val_score(cls,x,y,cv=cv))
        score2 = np.mean(cross_val_score(cls2,x,y,cv=cv))
        #print np.mean(cross_val_score(cls,x,y,cv=cv))
        neighbor_perf.append(score2)
        neighbor_perf_man.append(score)

plt.plot(neighbor_perf,label='euclidean')
plt.plot(neighbor_perf_man,label='manhattan')
plt.title('10-fold Cross-Validated Accuracy vs Neighbor Number')
plt.xlabel('Number of Neighbors')
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