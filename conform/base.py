import numpy as np

class CPBase:
    def p_val(self, scores):
        return sum([1 for s in scores if s >= scores[-1]])\
             / len(scores)

    def format(self, X, y = None):
        X = self.__list_to_ndarray(X)
        X = self.__reshape_if_vector(X)

        if y is not None:
            y = self.__list_to_ndarray(y)
            y = self.__reshape_if_scalar(y)
            return X, y

        return X

    def __list_to_ndarray(self, z):
        return np.array(z) if type(z) is list else z

    def __reshape_if_vector(self, X):
        return X.reshape(1, X.shape[0]) \
            if len(X.shape) == 1 else X

    def __reshape_if_scalar(self, y):
        return np.array([y]) if type(y) is not np.ndarray \
            else y
