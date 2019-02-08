from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import time
from gatherData import Dataset2

data = Dataset2()

x,y = data.fetch_data()
del data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)


print len(X_train)
print len(y_train)
#print y_train


cls = svm.SVC(kernel='linear')
#cls = svm.SVC(kernel='rbf')
#cls = svm.SVC(kernel='poly')
print ("TRAINING MODEL")
cls.fit(X_train,y_train.ravel())
print "Done training"

"""for t in range(0,len(y_test)):
 guess = []
 guess.append(X_test[t])
 guess = np.asarray(guess)
 print cls.predict(guess), " ---> " , y_test[t]"""

inf = cls.predict(X_test)
gt = y_test.ravel()

correct = 0

for t in range(len(gt)):
    if inf[t] == gt[t]:
        correct += 1

print correct / (len(gt) * 1.0)