import numpy as np
import math

from infinity import inf

from nc1nn import nn

def test_rust_nn():
    X = np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1]])
    y = np.array([0, 1, 2])
    x_ = np.array([3.0, 3.1])
    y_ = 3

    d_eq, d_neq, d_map = nn(X, y, x_, y_)

    assert d_eq == inf
    assert d_neq == math.sqrt(2.0)

    assert d_map[0][0] == False
    assert d_map[0][1] == math.sqrt(18.0)

    assert d_map[1][0] == False
    assert d_map[1][1] == math.sqrt(8.0)

    assert d_map[2][0] == False
    assert d_map[2][1] == math.sqrt(2.0)
