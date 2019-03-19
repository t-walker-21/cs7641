from sklearn.cluster import KMeans
fin = open("data1.txt","r")


X = []
y = []

for l in fin:
    X.append(l.split(",")[:-2])
    y.append(int(l.split(",")[-2]))


#print len(X)
#print len(y)



km = KMeans(n_clusters=3,random_state=0).fit(X[:-100])


print km.predict(X[0:2])
print y[0:2]