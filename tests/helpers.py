import numpy as np
np.random.seed(42)

from sklearn.datasets import load_iris

def __random_iris_sets():
    X, y = load_iris(True)

    idxs = np.arange(len(X))
    np.random.shuffle(idxs)

    X, y = X[idxs], y[idxs]

    X_train, y_train = X[:100], y[:100]
    X_test,  y_test  = X[100:], y[100:]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = __random_iris_sets()
