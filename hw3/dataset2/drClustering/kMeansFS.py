from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
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

scaler = MinMaxScaler()
scaler.fit(X)



X = scaler.transform(X)
dr_X = SelectKBest(chi2,k=2).fit_transform(X,y)

#obtain elbow plot
plot_elbow(dr_X)

#pick three clusters, and view a few groupings

km = KMeans(n_clusters=2,random_state=0).fit(dr_X)
