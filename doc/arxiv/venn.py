import numpy as np

from libconform import Venn
from libconform.vtx import VTXKNearestNeighbors

from sklearn.datasets import load_iris

X, y = load_iris(True)

# randomly permute X, y
indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X_train, y_train = X[:-20],    y[:-20]
X_test,  y_test  = X[-20:],    y[-20:]

vtx  = VTXKNearestNeighbors(n_neighbors=1)
venn = Venn(vtx)

venn.train(X_train, y_train)
res = venn.score(X_test, y_test)
print(res)
