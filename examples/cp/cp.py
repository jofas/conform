import numpy as np

from libconform import CP
from libconform.ncs import NCSDecisionTree, NCSKNearestNeighbors

from sklearn.datasets import load_iris

X, y = load_iris(True)

# randomly permute X, y
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X_train, y_train = X[:-20], y[:-20]
X_test,  y_test  = X[-20:], y[-20:]

epsilons = [0.01,0.025,0.05,0.1]

# offline decision tree
ncs = NCSDecisionTree()
cp = CP(ncs, epsilons)

cp.train(X_train, y_train)
res = cp.score(X_test, y_test)
print(res)

# online decision tree
ncs = NCSDecisionTree()
cp = CP(ncs, epsilons)

res = cp.score_online(X, y)
print(res)

# offline nearest neighbors
ncs = NCSKNearestNeighbors(n_neighbors=1)
cp = CP(ncs, epsilons)

cp.train(X_train, y_train)
res = cp.score(X_test, y_test)
print(res)

# online nearest neighbors
ncs = NCSKNearestNeighbors(n_neighbors=1)
cp = CP(ncs, epsilons)

cp.score_online(X, y)
print(res)
