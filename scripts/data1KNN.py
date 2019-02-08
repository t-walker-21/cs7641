import cv2
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time


mnist = fetch_mldata('MNIST original')

x = []
y = []
 
for i in range(0,len(mnist.target)):

 x.append(mnist.data[i])
 y.append(int(mnist.target[i]))



x = np.asarray(x)
y = np.asarray(y)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


cls = KNeighborsClassifier(n_neighbors=4)

cls.fit(X_train,y_train)

mat = confusion_matrix(cls.predict(X_test),y_test)


print(mat)


plt.matshow(mat)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


