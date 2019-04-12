from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

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
transformer = random_projection.GaussianRandomProjection(n_components=2)
dr_X = transformer.fit_transform(X)

#obtain elbow plot
#plot_elbow(dr_X)

#pick three clusters, and view a few groupings

km = KMeans(n_clusters=2,random_state=0).fit(dr_X)


labels = km.predict(dr_X)

print silhouette_score(dr_X,labels)

#show some results of each cluster
print "cluster 1"
print dr_X[np.where(labels == 1)][:5]

print "cluster 2"
print dr_X[np.where(labels == 0)][:5]