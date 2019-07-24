import numpy as np
np.random.seed(42)

from libconform.ncm import KNeighborsMean

from helpers import X_train, y_train, X_test, y_test

def test_knn_mean():
    X_train_ = X_train[:15,0].reshape(-1,1)
    X_test_  = X_test[:5,0].reshape(-1,1)
    y_train_, y_test_ = y_train[:15], y_test[:5]

    ncm = KNeighborsMean(n_neighbors=2)
    ncm.fit(X_train_, y_train_)

    scores = ncm.scores(X_test_, y_test_, X_eq_fit=False)
    print(scores)
    for score, true in zip(scores, [0.0,0.0,0.0,0.3,0.7]):
        assert round(score, 1) == true

def test_knn_mean_with_all_labels_per_observation():
    X_train_ = X_train[:15,0].reshape(-1,1)
    X_test_  = X_test[:5,0].reshape(-1,1)
    y_train_ = y_train[:15]

    ncm = KNeighborsMean(n_neighbors=5)
    ncm.fit(X_train_, y_train_)

    for x in X_test_:
        y = np.array([0, 1, 2])
        X = np.array([x for _ in y])
        scores = ncm.scores(X, y, X_eq_fit=False)
        assert sum(scores) >= 1.0

def test_knn_mean_x_eq_fit():
    pass

def test_knn_mean_unfitted():
    assert True == False

def test_knn_mean_k_bigger_training_data():
    assert True == False
