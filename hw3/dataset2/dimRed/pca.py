from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


X = load_wine().data
y = load_wine().target

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
pca = PCA(n_components=4)
pca.fit(X)
dr_X = pca.transform(X)

newData = [[],[],[]]
i = 0

for pt in dr_X:
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

print (pca.explained_variance_ratio_)