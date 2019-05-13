from . import NC1NN, NC1NN_py
from .helpers import load_usps_random

def test_rs_update_complete(benchmark):
    X, y = load_usps_random()
    nn = NC1NN()
    benchmark(nn.update_seq, X, y)

def test_rs_scores_complete(benchmark):
    X, y = load_usps_random()
    x_ = X[-1]
    y_ = y[-1]
    X = X[:-1]
    y = y[:-1]
    nn = NC1NN()
    nn.update_par(X, y)
    benchmark(nn.scores_seq, x_, y_)

def test_rs_par_update_complete(benchmark):
    X, y = load_usps_random()
    nn = NC1NN()
    benchmark(nn.update_par, X, y)

def test_rs_par_scores_complete(benchmark):
    X, y = load_usps_random()
    x_ = X[-1]
    y_ = y[-1]
    X = X[:-1]
    y = y[:-1]
    nn = NC1NN()
    nn.update_par(X, y)
    benchmark(nn.scores_par, x_, y_)
