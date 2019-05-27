import numpy as np

def append(X, y, X_, y_):
    if len(y.shape) > 1:
        return np.vstack((X, X_)), np.vstack((y, y_))
    else:
        return np.vstack((X, X_)), np.append(y, y_)

def format(X, y = None):
    X = __list_to_ndarray(X)
    X = __reshape_if_vector(X)

    if y is not None:
        y = __list_to_ndarray(y)
        y = __reshape_if_scalar(y)
        return X, y

    return X

def __list_to_ndarray(z):
    return np.array(z) if type(z) is list else z

def __reshape_if_vector(X):
    return X.reshape(1, X.shape[0]) \
        if len(X.shape) == 1 else X

def __reshape_if_scalar(y):
    return np.array([y]) if type(y) is not np.ndarray \
        else y
