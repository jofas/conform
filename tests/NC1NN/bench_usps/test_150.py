from . import NC1NN, NC1NN_py
from .helpers import load_usps_random

def load150():
    X, y = load_usps_random()
    return X[:150], y[:150]

def test_py_train_150(benchmark):
    X, y = load150()
    nn = NC1NN_py()
    benchmark(nn.train, X, y)

def test_py_scores_150(benchmark):
    X, y = load150()
    x_ = X[149]
    y_ = y[149]
    X = X[:149]
    y = y[:149]
    nn = NC1NN_py()
    nn.train(X, y)
    benchmark(nn.scores, x_, y_)

def test_rs_train_150(benchmark):
    X, y = load150()
    nn = NC1NN()
    benchmark(nn.train_seq, X, y)

def test_rs_scores_150(benchmark):
    X, y = load150()
    x_ = X[149]
    y_ = y[149]
    X = X[:149]
    y = y[:149]
    nn = NC1NN()
    nn.train_par(X, y)
    benchmark(nn.scores_seq, x_, y_)

def test_rs_par_train_150(benchmark):
    X, y = load150()
    nn = NC1NN()
    benchmark(nn.train_par, X, y)

def test_rs_par_scores_150(benchmark):
    X, y = load150()
    x_ = X[149]
    y_ = y[149]
    X = X[:149]
    y = y[:149]
    nn = NC1NN()
    nn.train_par(X, y)
    benchmark(nn.scores_par, x_, y_)
