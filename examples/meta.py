import numpy as np

from libconform import Meta, CP
from libconform.ncs import NCSKNearestNeighbors

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(True)

# randomly permute X, y
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X_train, y_train = X[:-20],    y[:-20]
X_test,  y_test  = X[-20:],    y[-20:]

epsilons = [0.01,0.025,0.05,0.1]

B         = KNeighborsClassifier(n_neighbors=1)
B_train   = lambda X, y: B.fit(X,y)
B_predict = lambda X: B.predict(X)

ncs       = NCSKNearestNeighbors(n_neighbors=1)
M         = CP(ncs, [])
M_train   = lambda X, y: M.train(X, y)
M_predict = lambda X: M.p_vals(X)

clf = Meta(M_train, M_predict, B_train, B_predict, epsilons)

clf.train(X_train, y_train, k_folds = 10)

res = clf.score(X_test, y_test)
print(res)
