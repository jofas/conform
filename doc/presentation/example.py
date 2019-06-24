import numpy as np
from libconform import CP
from libconform.ncs import NCSKNearestNeighbors

from sklearn.datasets import load_iris

X, y = load_iris(True)

# randomly permute X, y
indices = np.arange(len(X))
np.random.shuffle(indices)

X, y = X[indices], y[indices]

# split in train and test data set
X_train, y_train = X[:-20], y[:-20]
X_test,  y_test  = X[-20:], y[-20:]

ncs = NCSKNearestNeighbors(n_neighbors=1)
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
cp = CP(ncs, epsilons)

cp.train(X_train, y_train)

res = cp.score(X_test, y_test)
print(res)
