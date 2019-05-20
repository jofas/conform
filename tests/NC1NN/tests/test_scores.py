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

    score_test = {0: 0.5, 1: 2.0}

    score = nn.score(x_, list(range(2)))

    for i in range(2):
        assert score[i] == score_test[i]

def test_score_empty():
    nn = NC1NN()
    assert nn.score(np.array([0.0]), [0])[0,0] == 0.0
