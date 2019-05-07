import math
import numpy as np

from copy import copy
from infinity import inf

class NC1NN:
    def __init__(self):
        self.X       = np.array([])
        self.y       = np.array([])
        self.scores_ = []
        self.dists   = []

    def update(self, X, y):
        for i in range(X.shape[0]):
            self.__update(X[i], y[i])

    def scores(self, x_, y_):
        scores = np.zeros(self.X.shape[0] + 1)
        d_eq, d_neq, d_map = self.nn(x_, y_)

        for i in range(self.X.shape[0]):
            scores[i] = self.scores_[i]

            neq, d = d_map[i]
            if neq:
                if self.dists[i][1] > d:
                    eq_d = self.dists[i][0]; neq_d = d
                    scores[i] = self.__score(eq_d, neq_d)
            else:
                if self.dists[i][0] > d:
                    eq_d = d; neq_d = self.dists[i][1]
                    scores[i] = self.__score(eq_d, neq_d)

        scores[-1] = self.__score(d_eq, d_neq)
        return scores

    def __update(self, x_, y_):
        d_eq, d_neq, d_map = self.nn(x_, y_)

        # update
        for i in range(self.X.shape[0]):
            neq, d = d_map[i]
            if neq:
                if self.dists[i][1] > d:
                    self.dists[i][1] = d
                    self.scores_[i] = \
                        self.__score(*self.dists[i])
            else:
                if self.dists[i][0] > d:
                    self.dists[i][0] = d
                    self.scores_[i] = \
                        self.__score(*self.dists[i])
        # put
        if self.X.shape[0] == 0: self.X = np.array([x_])
        else: self.X = np.vstack((self.X, x_))
        self.y = np.append(self.y, y_)

        self.dists.append([d_eq, d_neq])
        self.scores_.append(self.__score(d_eq,d_neq))

    def nn(self, x_, y_):
        d_eq, d_neq = inf, inf
        d_map = []

        for i in range(self.X.shape[0]):
            d   = np.linalg.norm(x_ - self.X[i])
            neq = self.y[i] != y_

            d_map.append([neq, d])

            if neq:
                if d < d_neq: d_neq = d
            else:
                if d < d_eq: d_eq = d

        return d_eq, d_neq, np.array(d_map)

    def __score(self, eq_d, neq_d):
        return 0.0          if eq_d == neq_d else \
               eq_d / neq_d if neq_d > 0.0   else \
               inf

class NC1NN_old:
    def __init__(self):
        self.X      = np.array([])
        self.y      = np.array([])
        #self.scores = np.array([])

        #self.nns    = np.array([])

    # not dynamic
    # add to the data sets and compute new scores
    def update(self, X, y):
        #self.scores = self.non_conformity(x, y)
        if self.X.shape[0] == 0: self.X = np.array(X)
        else: self.X = np.vstack((self.X, X))
        self.y = np.append(self.y, y)

    # not dynamic
    # return scores for every element in (X + x) without
    # updating the scores
    def scores(self, x, y):
        scores = np.zeros(self.X.shape[0] + 1)
        # for each element compute score
        for i in range(self.X.shape[0] + 1):
            cx, cy = self.__elem_by_idx(x, y, i)

            nn_eq_d  = inf
            nn_neq_d = inf

            # in order to do that find the two nearest
            # neighbors
            for j in range(self.X.shape[0] + 1):
                # do not include self in scoring
                if j == i: continue

                nx, ny = self.__elem_by_idx(x, y, j)

                d = np.linalg.norm(cx - nx)

                if cy == ny and d < nn_eq_d : nn_eq_d  = d
                if cy != ny and d < nn_neq_d: nn_neq_d = d

            # now compute nc value and put it in scores
            scores[i] = self.__score(nn_eq_d, nn_neq_d)

        return scores

    def __elem_by_idx(self, x, y, idx):
        s = self.X.shape[0]
        if idx == s: return x, y
        return self.X[idx], self.y[idx]

    def __score(self, eq_d, neq_d):
        return 0.0          if eq_d == neq_d else \
               eq_d / neq_d if neq_d > 0.0   else \
               inf

