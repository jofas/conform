import numpy as np

from infinity import inf

from . import NC1NN
from .helpers import vec_cmp

def test_scores():
    X = np.array([[0.0],[1.0],[2.0],[3.0]])
    y = np.array([0, 0, 1, 0])

    x_ = np.array([4.0])

    nn = NC1NN()
    nn.train(X, y)

    scores_test = {
        0: np.array([0.5, 0.0, inf, 0.0, 0.5]),
        1: np.array([0.5, 0.0, 2.0, 2.0, 2.0]),
    }

    for i in range(2):
        scores = nn.scores(x_, i)
        vec_cmp(scores, scores_test[i])

def test_scores_empty():
    nn = NC1NN()
    assert nn.scores(np.array([0.0]), 0)[0] == 0.0
