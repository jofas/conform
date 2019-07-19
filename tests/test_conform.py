import numpy as np
np.random.seed(42)

from libconform import CP
from libconform.ncs import NCSKNearestNeighbors

from sklearn.datasets import load_iris

def test_predict():
    X, y = load_iris(True)

    idxs = np.arange(len(X))
    np.random.shuffle(idxs)

    X, y = X[idxs], y[idxs]

    X_train, y_train = X[:100], y[:100]
    X_test,  y_test  = X[100:], y[100:]

    sig_lvls = [0.01, 0.025, 0.05, 0.1]
    ncs = NCSKNearestNeighbors()
    cp = CP(ncs, sig_lvls)

    cp.train(X_train, y_train)

    pred = cp.predict(X_test)
    print(pred)
    print(cp.labels.reverse_map)
    assert True == False
