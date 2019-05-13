from . import NC1NN, NC1NN_py
from .helpers import load_usps_random

def load150():
    X, y = load_usps_random()
    return X[:150], y[:150]

def test_py_update_150(benchmark):
    X, y = load150()
    nn = NC1NN_py()
    benchmark(nn.update, X, y)

def test_py_scores_150(benchmark):
    X, y = load150()
    x_ = X[149]
    y_ = y[149]
    X = X[:149]
    y = y[:149]
    nn = NC1NN_py()
    nn.update(X, y)
    benchmark(nn.scores, x_, y_)

def test_rs_update_150(benchmark):
    X, y = load150()
    nn = NC1NN()
    benchmark(nn.update_seq, X, y)

def test_rs_scores_150(benchmark):
    X, y = load150()
    x_ = X[149]
    y_ = y[149]
    X = X[:149]
    y = y[:149]
    nn = NC1NN()
    nn.update_par(X, y)
    benchmark(nn.scores_seq, x_, y_)

def test_rs_par_update_150(benchmark):
    X, y = load150()
    nn = NC1NN()
    benchmark(nn.update_par, X, y)

def test_rs_par_scores_150(benchmark):
    X, y = load150()
    x_ = X[149]
    y_ = y[149]
    X = X[:149]
    y = y[:149]
    nn = NC1NN()
    nn.update_par(X, y)
    benchmark(nn.scores_par, x_, y_)
