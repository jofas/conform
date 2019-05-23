import numpy as np

from infinity import inf
from sklearn.neighbors import NearestNeighbors as NN

from .base import NCSBase

class _LabelNN:
    def __init__(self, n, label, **sklearn):
        self.label = label
        self.clf   = NN(**sklearn)
        self.n_set = n
        self.n_cur = 0

    def train(self, X):
        if X.shape[0] < self.n_set: self.n_cur = X.shape[0]
        else: self.n_cur = self.n_set
        if self.n_cur > 0: self.clf.fit(X)

    def nn(self, x):
        if self.n_cur > 0:
            return self.clf.kneighbors(x, self.n_cur)[0][0]
        else:
            return np.array([inf], dtype=np.float64)

class NCSKNearestNeighbors(NCSBase):
    def __init__(self, labels, **sklearn):
        self.n =  5 if "n_neighbors" not in sklearn \
            else sklearn["n_neighbors"]

        # cp: ignore first nn, icp: ignore last nn
        self.n += 1

        self.clfs = {k: _LabelNN(self.n, k, **sklearn) \
            for k in labels}

    def train(self, X, y):
        split = {k: np.array([]) for k in self.clfs}

        for x_, y_ in zip(X, y):
            if split[y_].shape[0] == 0:
                split[y_] = x_.reshape(1,-1)
            else:
                split[y_] = np.vstack((split[y_], x_))

        for k in split:
            self.clfs[k].train(np.array(split[k]))

    def scores(self, X, y, cp):
        return [self.__score(x_.reshape(1,-1), y_, cp) \
            for x_, y_ in zip(X, y)]

    def score(self, x, labels):
        return [self.__score(x.reshape(1,-1), y) \
            for y in labels]

    def __score(self, x, y, contains_x = False):
        nns = {k: self.clfs[k].nn(x) for k in self.clfs}

        dists_neq = np.array([])
        for k in nns:
            if k == y: continue
            if dists_neq.shape[0] == 0: dists_neq = nns[k]
            else: dists_neq = np.append(dists_neq, nns[k])

        n = self.n -1 if self.n - 1 <= len(dists_neq) \
                else len(dists_neq)

        d_neq = sum(sorted(dists_neq)[:n])

        d_eq = sum(nns[y][1:]) if contains_x else \
            sum(nns[y][:-1])

        return 0.0          if d_eq == d_neq else \
               d_eq / d_neq if d_neq > 0.0   else \
               inf
