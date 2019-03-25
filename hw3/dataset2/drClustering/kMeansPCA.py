from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    plt.title('The Elbow Method showing the optimal k')
    plt.show()




X = load_wine().data
y = load_wine().target

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
pca = PCA(n_components=2)
pca.fit(X)
dr_X = pca.transform(X)

#obtain elbow plot
plot_elbow(dr_X)

#pick three clusters, and view a few groupings

km = KMeans(n_clusters=3,random_state=0).fit(dr_X)

labels = km.predict(dr_X)

print silhouette_score(dr_X,labels)

#show some results of each cluster
print "cluster 1"
print dr_X[np.where(labels == 1)][:5]

print "cluster 2"
print dr_X[np.where(labels == 0)][:5]