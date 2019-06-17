import numpy as np

from libconform import ICP
from libconform.ncs import NCSDecisionTree, NCSKNearestNeighbors

from sklearn.datasets import load_iris

X, y = load_iris(True)

# randomly permute X, y
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X_train, y_train = X[:-50],    y[:-50]
X_cal,   y_cal   = X[-50:-20], y[-50:-20]
X_test,  y_test  = X[-20:],    y[-20:]

epsilons = [0.01,0.025,0.05,0.1]

# decision tree
ncs = NCSDecisionTree()
icp = ICP(ncs, epsilons)

icp.train(X_train, y_train)
icp.calibrate(X_cal, y_cal)
res = icp.score(X_test, y_test)
print(res)

# nearest neighbors
ncs = NCSKNearestNeighbors(n_neighbors=1)
icp = ICP(ncs, epsilons)

icp.train(X_train, y_train)
icp.calibrate(X_cal, y_cal)
res = icp.score(X_test, y_test)
print(res)
