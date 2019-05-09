import numpy as np

from infinity import inf

from . import NC1NN

def test_scores():
    X_new = np.array([[0.0],[1.0],[2.0],[3.0],[4.0]])
    y_new = np.array([0, 0, 1, 0, 1])

    nn = NC1NN()
    nn.update(X_new, y_new)

    dists, scores = nn.dists, nn.scores_

    dists_test  = np.array( [ [1.0, 2.0]
                            , [1.0, 1.0]
                            , [2.0, 1.0]
                            , [2.0, 1.0]
                            , [2.0, 1.0] ] )

    #TODO: secure score like update

    scores_test = np.array([0.5, 0.0, 2.0, 2.0, 2.0])

    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            assert dists[i,j] == dists_test[i,j]
        assert scores[i] == scores_test[i]

