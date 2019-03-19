from sklearn import random_projection
fin = open("data1.txt","r")

X = []
y = []

for l in fin:
    X.append(l.split(",")[:-2])
    y.append(int(l.split(",")[-2]))

pca = PCA(n_components=3)
pca.fit(X)

print (pca.explained_variance_ratio_)