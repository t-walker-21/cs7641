from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

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

#print len(X[0])

#obtain elbow plot
plot_elbow(X)

#pick three clusters, and view a few groupings

km = KMeans(n_clusters=3,random_state=0).fit(X)

cent0 = km.cluster_centers_[0]
cent1 = km.cluster_centers_[1]
cent2 = km.cluster_centers_[2]

centGroups = [[],[],[]]


for pt in X:
    #print pt
    distances = []
    distances.append(np.linalg.norm(cent0-pt))
    distances.append(np.linalg.norm(cent1-pt))
    distances.append(np.linalg.norm(cent2-pt))
    closest = np.argmin(np.array(distances))
    centGroups[closest].append(pt)

#print centGroups[2]