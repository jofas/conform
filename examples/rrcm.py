import numpy as np

from libconform import RRCM
from libconform.ncs import NCSKNearestNeighborsRegressor

from sklearn.datasets import load_boston

X, y = load_boston(True)

# randomly permute X, y
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X_train, y_train = X[:-20],    y[:-20]
X_test,  y_test  = X[-20:],    y[-20:]

epsilons = [0.01,0.025,0.05,0.1]

# online
ncs  = NCSKNearestNeighborsRegressor(n_neighbors=1)
rrcm = RRCM(ncs, epsilons)

res = rrcm.score_online(X, y)
print(res)

# offline
ncs  = NCSKNearestNeighborsRegressor(n_neighbors=1)
rrcm = RRCM(ncs, epsilons)

rrcm.train(X_train, y_train)
res = rrcm.score(X_test, y_test)
print(res)
