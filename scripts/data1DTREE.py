import cv2
from sklearn.datasets import fetch_mldata
from sklearn import tree
from sklearn.cross_validation import train_test_split
import numpy as np

mnist = fetch_mldata('MNIST original')

x = []
y = []
 
for i in range(0,len(mnist.target)):

 x.append(mnist.data[i])
 y.append(int(mnist.target[i]))



x = np.asarray(x)
y = np.asarray(y)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.75, random_state=42)


x = np.asarray(x)
y = np.asarray(y)

cls = tree.DecisionTreeClassifier()

cls.fit(x,y)


for t in range(0,len(y_test)):
 guess = []
 guess.append(X_test[t])
 guess = np.asarray(guess)

 cv2.imshow(str(cls.predict(guess)[0]),X_test[t].reshape(28,28))
 cv2.waitKey(700)
