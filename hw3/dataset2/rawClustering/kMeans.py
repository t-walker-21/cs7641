from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
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
    plt.title('KMeans Elbow on Wine Quality Data')
    plt.show()


X = load_wine().data

#print len(X[0])

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