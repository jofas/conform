import h5py
import numpy as np

from .. import NC1NN
from .nc1nn import nn, nn_par

def usps_test_set(amount):
    with h5py.File('usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]

        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]


    X = np.vstack((X_tr, X_te))
    y = np.concatenate((y_tr, y_te))

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    x_ = X[amount - 1]
    y_ = y[amount - 1]
    X  = X[:amount - 1]
    y  = y[:amount - 1]

    return X, y, x_, y_

'''
def test_usps_rs_50(benchmark):
    X, y, x_, y_ = usps_test_set(50)
    benchmark(nn, X, y, x_, y_)

def test_usps_rs_par_50(benchmark):
    X, y, x_, y_ = usps_test_set(50)
    benchmark(nn, X, y, x_, y_)

def test_usps_py_50(benchmark):
    X, y, x_, y_ = usps_test_set(50)

    nc = NC1NN()
    nc.update(X, y)

    benchmark(nc.nn, x_, y_)


def test_usps_rs_100(benchmark):
    X, y, x_, y_ = usps_test_set(100)
    benchmark(nn, X, y, x_, y_)

def test_usps_rs_par_100(benchmark):
    X, y, x_, y_ = usps_test_set(100)
    benchmark(nn, X, y, x_, y_)

def test_usps_py_100(benchmark):
    X, y, x_, y_ = usps_test_set(100)

    nc = NC1NN()
    nc.update(X, y)

    benchmark(nc.nn, x_, y_)


def test_usps_rs_150(benchmark):
    X, y, x_, y_ = usps_test_set(150)
    benchmark(nn, X, y, x_, y_)

def test_usps_rs_par_150(benchmark):
    X, y, x_, y_ = usps_test_set(150)
    benchmark(nn, X, y, x_, y_)

def test_usps_py_150(benchmark):
    X, y, x_, y_ = usps_test_set(150)

    nc = NC1NN()
    nc.update(X, y)

    benchmark(nc.nn, x_, y_)

def test_usps_rs_300(benchmark):
    X, y, x_, y_ = usps_test_set(300)
    benchmark(nn, X, y, x_, y_)
'''

def test_usps_rs_par_300(benchmark):
    from copy import deepcopy
    X, y, x_, y_ = usps_test_set(9000)

    for i in range(7):
        X_  = deepcopy(X)
        y__ = deepcopy(y)
        X = np.vstack((X, X_))
        y = np.append(y, y__)

    benchmark(nn, X, y, x_, y_)

