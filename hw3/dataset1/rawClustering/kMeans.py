from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


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
    plt.title('KMeans Elbow on Grad Admissions Data')
    plt.show()



fin = open("../data1.txt","r")


X = []
y = []

for l in fin:
    X.append(l.split(",")[:-2])
    y.append(int(l.split(",")[-2]))

X = np.array(X,dtype=np.float32)

#obtain elbow plot
plot_elbow(X)

#pick three clusters, and view a few groupings

km = KMeans(n_clusters=2,random_state=10).fit(X)
labels = km.predict(X)

print silhouette_score(X,labels)

#show some results of each cluster
print "cluster 1"
print X[np.where(labels == 1)][:5]

print "cluster 2"
print X[np.where(labels == 0)][:5]
