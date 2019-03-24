from sklearn import mixture
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_bic(X):

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                        covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                    (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    plt.xlabel('Number of components')
    plt.legend([b[0] for b in bars], cv_types)

    plt.show()


fin = open("../data1.txt","r")


X = []
y = []

for l in fin:
    X.append(l.split(",")[:-2])
    y.append(int(l.split(",")[-2]))

X = np.array(X,dtype=np.float32)

X = np.array(X,dtype=np.float32)
scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
pca = PCA(n_components=2)
pca.fit(X)
dr_X = pca.transform(X)

plot_bic(X)

gmm = mixture.GaussianMixture(n_components=2,covariance_type='tied')
gmm.fit(X)


gaussianGroups = [[],[]]

for pt in X:
    res = gmm.predict(pt.reshape(1,-1))[0]
    gaussianGroups[res].append(pt)

print gaussianGroups[1]