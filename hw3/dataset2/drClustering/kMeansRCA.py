from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection
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
y = load_wine().target

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
transformer = random_projection.GaussianRandomProjection(n_components=2)
dr_X = transformer.fit_transform(X)

#obtain elbow plot
plot_elbow(dr_X)

#pick three clusters, and view a few groupings

km = KMeans(n_clusters=2,random_state=0).fit(dr_X)

cent0 = km.cluster_centers_[0]
cent1 = km.cluster_centers_[1]



'''for pt in X:
    #print pt
    distances = []
    distances.append(np.linalg.norm(cent0-pt))
    distances.append(np.linalg.norm(cent1-pt))
    distances.append(np.linalg.norm(cent2-pt))
    closest = np.argmin(np.array(distances))
    centGroups[closest].append(pt)

print centGroups[2]'''