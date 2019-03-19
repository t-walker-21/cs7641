from sklearn import mixture
fin = open("data1.txt","r")


X = []
y = []

for l in fin:
    X.append(l.split(",")[:-2])
    y.append(int(l.split(",")[-2]))


gmm = mixture.GaussianMixture(n_components=3)
gmm.fit(X)

