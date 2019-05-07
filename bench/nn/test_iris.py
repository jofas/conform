from sklearn.datasets import load_iris

import numpy as np

from .. import NC1NN
from .nc1nn import nn

def iris_test_set(amount):
    iris = load_iris()

    X = iris['data']
    y = iris['target']

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    x_ = X[amount - 1]
    y_ = y[amount - 1]
    X  = X[:amount - 1]
    y  = y[:amount - 1]

    return X, y, x_, y_

def test_iris_rs_50(benchmark):
    X, y, x_, y_ = iris_test_set(50)
    benchmark(nn, X, y, x_, y_)

def test_iris_py_50(benchmark):
    X, y, x_, y_ = iris_test_set(50)

    nc = NC1NN()
    nc.update(X, y)

    benchmark(nc.nn, x_, y_)

def test_iris_rs_100(benchmark):
    X, y, x_, y_ = iris_test_set(100)
    benchmark(nn, X, y, x_, y_)

def test_iris_py_100(benchmark):
    X, y, x_, y_ = iris_test_set(100)

    nc = NC1NN()
    nc.update(X, y)

    benchmark(nc.nn, x_, y_)

def test_iris_rs_150(benchmark):
    X, y, x_, y_ = iris_test_set(150)
    benchmark(nn, X, y, x_, y_)

def test_iris_py_150(benchmark):
    X, y, x_, y_ = iris_test_set(150)

    nc = NC1NN()
    nc.update(X, y)

    benchmark(nc.nn, x_, y_)
