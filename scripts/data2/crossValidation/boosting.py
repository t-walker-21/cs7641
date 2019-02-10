from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn import tree
import numpy as np
from cleanData import Dataset1
from matplotlib import pyplot as plt

data = Dataset1()

x,y = data.fetch_data_multi()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=20)


print len(X_train)
print len(y_train)




est_perf = []
for est in range (1,10):
        cls = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2),n_estimators=est)

        cv = ShuffleSplit(n_splits=10, test_size=0.2,random_state=0)
        score = cross_val_score(cls,x,y,cv=cv)
        #print np.mean(cross_val_score(cls,x,y,cv=cv))
        est_perf.append(np.mean(score))

plt.plot(est_perf)
plt.title('10-fold Cross-Validated Accuracy vs Boosted Trees of Depth 2')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()