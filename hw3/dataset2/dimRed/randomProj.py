from sklearn import random_projection
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

X = load_wine().data
y = load_wine().target

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)

print X

transformer = random_projection.GaussianRandomProjection(n_components=2)
X_new = transformer.fit_transform(X)

newData = [[],[],[]]
i = 0

for pt in X_new:
    newData[y[i]].append(pt)
    i += 1


print np.array(newData[0]).shape
for i in range(len(newData[0])):
    plt.scatter(newData[0][i][0],newData[0][i][1],marker='+',color='r')
    
for i in range(len(newData[1])):
    plt.scatter(newData[1][i][0],newData[1][i][1],marker='+',color='g')

for i in range(len(newData[2])):
    plt.scatter(newData[2][i][0],newData[2][i][1],marker='+',color='b')

plt.show()