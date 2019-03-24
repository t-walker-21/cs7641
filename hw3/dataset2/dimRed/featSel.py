from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

X = load_wine().data
y = load_wine().target

scaler = MinMaxScaler()
scaler.fit(X)

X = scaler.transform(X)

X_new = SelectKBest(chi2,k=2).fit_transform(X,y)

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

plt.title('Select K best features of Wine Quality')
plt.xlabel('first projection')
plt.ylabel('second projection')
plt.show()