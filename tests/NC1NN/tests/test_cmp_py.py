import numpy as np

from sklearn.datasets import load_iris

from . import NC1NN, NC1NN_py, CP
from .helpers import vec_cmp

# Sample from IRIS data set. The same sample is used in
# A Tutorial on Conformal Prediction
IRIS_SUBSET_X = [ [5.0], [4.4], [4.9], [4.4], [5.1]
                , [5.9], [5.0], [6.4], [6.7], [6.2]
                , [5.1], [4.6], [5.0], [5.4], [5.0]
                , [6.7], [5.8], [5.5], [5.8], [5.4]
                , [5.1], [5.7], [4.6], [4.6], [6.8] ]

# setosa -> 0, versicolor -> 1
IRIS_SUBSET_Y = [ 0, 0, 0, 0, 0
                , 1, 0, 1, 1, 1
                , 0, 0, 0, 0, 1
                , 1, 1, 0, 1, 0
                , 0, 1, 0, 0, 1 ]

def test_scores_from_tutorial():
    X  = np.array(IRIS_SUBSET_X[:24])
    y  = np.array(IRIS_SUBSET_Y[:24])
    x_ = np.array(IRIS_SUBSET_X[24])
    y_ = IRIS_SUBSET_Y[24]

    nn = NC1NN()
    nn.train(X, y)
    s1 = nn.scores(x_, y_)

    nn = NC1NN_py()
    nn.train(X, y)
    s2 = nn.scores(x_, y_)

    vec_cmp(s1, s2)

def test_nc_from_tutorial():
    X  = np.array(IRIS_SUBSET_X[:24])
    y  = np.array(IRIS_SUBSET_Y[:24])
    x_ = np.array(IRIS_SUBSET_X[24])

    epsilons = [0.05, 0.08, 0.33]

    res = [{0.05: [0,1], 0.08: [1], 0.33:[]}]

    cp = CP(NC1NN(), epsilons, np.array([0,1]))
    cp.train(X, y)
    res1 = cp.predict(x_)

    cp = CP(NC1NN_py(), epsilons, np.array([0,1]))
    cp.train(X, y)
    res2 = cp.predict(x_)

    assert res == res1 == res2

def test_scores_from_random_iris_samples():
    iris = load_iris()

    X = iris['data']
    y = iris['target']

    indices = np.arange(X.shape[0])

    for i in range(20):
        np.random.shuffle(indices)

        X_  = X[indices][:-1]
        y_  = y[indices][:-1]
        x__ = X[indices][-1]
        y__ = y[indices][-1]

        nn = NC1NN()
        nn.train(X_, y_)
        s1 = nn.scores(x__, y__)

        nn = NC1NN_py()
        nn.train(X_, y_)
        s2 = nn.scores(x__, y__)

        vec_cmp(s1, s2)
