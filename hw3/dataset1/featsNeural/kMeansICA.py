from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA as ICA
import matplotlib.pyplot as plt
import os
from keras import applications
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from keras.utils import to_categorical


def plot_elbow(X):
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()



fin = open("../data1.txt","r")


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

dr_X = ica.transform(X)

#obtain elbow plot
plot_elbow(dr_X)

#pick three clusters, and view a few groupings

km = KMeans(n_clusters=4,random_state=0).fit(dr_X)

newX = []

for pt in dr_X:
    newX.append(km.predict(pt.reshape(1,-1))[0])

newX = np.array(newX)

newX = to_categorical(newX)
y = to_categorical(y)

print len(newX)


INIT_LR = 5E-4
EPOCHS = 500
BS = 50
seed = 9068

np.random.seed(seed)

print newX[0]


cls = Sequential()

cls.add(Dense(2,input_dim=4,activation='relu',kernel_initializer='random_uniform'))
cls.add(Dense(30,activation='relu',kernel_initializer='random_uniform'))
cls.add(Dense(3,activation='softmax',kernel_initializer='random_uniform'))

opt = Adam(lr=INIT_LR)

cls.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

history = cls.fit(newX,y,epochs=EPOCHS,steps_per_epoch=10,validation_split=0.2,validation_steps=50)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'train'], loc='upper left')
plt.show()