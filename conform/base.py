import numpy as np
import random

from .metrics import CPMetrics

class CPBase:
    def __init__(self, A, epsilons, labels, smoothed):
        self.A        = A
        self.epsilons = epsilons
        self.labels   = labels
        self.smoothed = smoothed

    def train(self, X, y):
        X, y = self.format(X, y)
        self.A.train(X, y)

    def predict(self, X):
        X = self.format(X)
        res = [ None for _ in range(X.shape[0]) ]

        for i in range(X.shape[0]):
            predicted = { e: [] for e in self.epsilons }

            scores = self.A.scores(X[i], self.labels)

            for (label, s) in zip(self.labels, scores):
                p = self.__p_val_smoothed(s) \
                    if self.smoothed else self.__p_val(s)

                for epsilon in self.epsilons:
                    if p > epsilon:
                        predicted[epsilon].append(label)

            res[i] = predicted

        return res

    def score(self, X, y):
        X, y = self.format(X, y)

        res = CPMetrics(self.epsilons)
        predicted = self.predict(X)

        for i in range(X.shape[0]):
            res.update(predicted[i], y[i])

        return res

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

    def __p_val_smoothed(self, scores):
        bigger = 0; eq = 0; s_ = scores[-1]
        for s in scores:
            if s >  s_: bigger += 1
            if s == s_: eq += 1
        return (bigger + random.uniform(0.0, 1.0) * eq) \
             / len(scores)

    def __p_val(self, scores):
        return sum([1 for s in scores if s >= scores[-1]])\
             / len(scores)
