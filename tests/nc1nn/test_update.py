import numpy as np

from infinity import inf

from nc1nn import update

def test_update():
    X_new = np.array([[1.0],[2.0],[3.0],[4.0]])
    y_new = np.array([0, 1, 0, 1])

    X_seen = np.array([[0.0]])
    y_seen = np.array([0])

    dists  = np.array([[inf, inf]], dtype=np.float64)
    scores = np.array([0.0])

    X, y, dists, scores = \
        update(X_new, y_new, X_seen, y_seen, dists, scores)

    X_test = np.vstack((X_seen, X_new))
    y_test = np.append(y_seen, y_new)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            assert X[i,j] == X_test[i,j]
        assert y[i] == y_test[i]

    dists_test  = np.array( [ [1.0, 2.0]
                            , [1.0, 1.0]
                            , [2.0, 1.0]
                            , [2.0, 1.0]
                            , [2.0, 1.0] ] )
    scores_test = np.array([0.5, 0.0, 2.0, 2.0, 2.0])
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            assert dists[i,j] == dists_test[i,j]
        assert scores[i] == scores_test[i]
