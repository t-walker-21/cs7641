from sklearn.decomposition import FastICA as ICA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

fin = open("../data1.txt","r")
import numpy as np

X = []
y = []

for l in fin:
    X.append(l.split(",")[:-2])
    y.append(int(l.split(",")[-2]))

X = np.array(X,dtype=np.float32)
scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)

ica = ICA(n_components=2)
ica.fit(X)

X = ica.transform(X)

newData = [[],[],[]]
i = 0

for pt in X:
    newData[y[i]].append(pt)
    i += 1


print np.array(newData[0]).shape
for i in range(len(newData[0])):
    plt.scatter(newData[0][i][0],newData[0][i][1],marker='+',color='r')
    
for i in range(len(newData[1])):
    plt.scatter(newData[1][i][0],newData[1][i][1],marker='+',color='g')

for i in range(len(newData[2])):
    plt.scatter(newData[2][i][0],newData[2][i][1],marker='+',color='b')

plt.title('Independent Components of Grad Admissions')
plt.xlabel('first component')
plt.ylabel('second component')
plt.show()