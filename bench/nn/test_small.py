# test nc_scores.NC1NN.nn vs nc1nn.nn with:
#   1. with small test like in tests/test_nn
#   2. with iris ( 50, 100, 150 ) randomly shuffled
#   3. with usps ( 50, 100, 150 ) randomly shuffled
import numpy as np

from .. import NC1NN
from .nc1nn import nn

def small_test_set():
    X = np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]])
    y = np.array([0, 1, 2])
    x_ = np.array([3.0, 3.1])
    y_ = 3

    return X, y, x_, y_

def test_small_rs(benchmark):
    X, y, x_, y_ = small_test_set()
    benchmark(nn, X, y, x_, y_)

def test_small_py(benchmark):
    X, y, x_, y_ = small_test_set()

    nc = NC1NN()
    nc.update(X, y)

    benchmark(nc.nn, x_, y_)
