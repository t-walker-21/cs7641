from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from cleanData import Dataset1
import numpy as np
import time

data = Dataset1()

x,y = data.fetch_data_multi()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=412)


print len(X_train)
print len(y_train)


#cls = KNeighborsClassifier(n_neighbors=1,metric="manhattan")
cls = KNeighborsClassifier(n_neighbors=2,metric="manhattan")

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

"""
plt.matshow(mat)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()"""