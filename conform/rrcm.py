import numpy as np

from infinity import inf
from statistics import mean, median

from sklearn.neighbors import NearestNeighbors as NN

from .metrics import RRCMMetrics
from .ncs.base import NCSBaseRegressor
from . import util

INVALID_A = "Non-conformity regressor invalid"

INTERVAL  = 0
RAY_UNION = 1
RAY_NEG   = 2
RAY       = 3
POINT     = 4
R         = 5
EMPTY     = 6

ZERO_VALS = [R, EMPTY]
ONE_VAL   = [RAY_NEG, RAY, POINT]
TWO_VALS  = [INTERVAL, RAY_UNION]

class RRCM:
    def __init__(self, A, epsilons, convex_hull = True):

        if NCSBaseRegressor not in type(A).__bases__:
            raise Exception(INVALID_A)

        self.A           = A
        self.epsilons    = epsilons
        self.convex_hull = convex_hull

        self.init_train  = False
        self.X_train     = None
        self.y_train     = None

        self.C           = []
        self.D           = []

    def train(self, X, y, override = False):
        X, y = util.format(X, y)

        if not override and self.init_train:
            self.X_train, self.y_train = util.append(
                self.X_train, self.y_train, X, y
            )
        else:
            self.X_train, self.y_train = X, y
            self.init_train            = True

        self.A.train(self.X_train, self.y_train)
        self.C, self.D = self.A.coeffs(X, y, cp = True)

    def predict(self, X):
        X = util.format(X)

        res = []
        for x in X:
            P, S = self.__ps(x)

            M, N0, N = self.__mn0n(P, S)

            res.append(
                self.__intervals(M, N0, N, P, len(S) + 1)
            )

        return res

    def score(self, X, y):
        X, y = util.format(X, y)

        res = RRCMMetrics(self.epsilons)

        predicted = self.predict(X)

        for p_, y_ in zip(predicted, y):
            res.update(p_, y_)

        return res

    def score_online(self, X, y):
        X, y = util.format(X, y)

        res = RRCMMetrics(self.epsilons)

        for x_, y_ in zip(X, y):
            p = self.predict(x_)[0]
            res.update(p, y_)

        return res

    def __ps(self, x):
        cn, dn = self.A.coeffs_n(x)
        if cn < 0.0: cn = -cn; dn = -dn

        bound0 = lambda ci, di: -(di - dn) / (ci - cn)
        bound1 = lambda ci, di: -(di + dn) / (ci + cn)
        bound2 = lambda ci, di: -(di + dn) / 2 * ci

        P = []; S = []
        for i, ci, di in zip(
            range(len(self.C)), self.C, self.D
        ):
            if ci < 0.0: ci = -ci; di = -di

            if cn == ci:
                if cn > 0.0:
                    ui = bound2(ci, di)

                    P.append((i, ui))

                    if dn > di:
                        S.append(_S(RAY_NEG, ui))
                    elif dn < di:
                        S.append(_S(RAY, ui))
                    else:
                        P.pop()
                        S.append(_S(R))
                else:
                    if abs(dn) <= abs(di):
                        S.append(_S(R))
                    else:
                        S.append(_S(EMPTY))
            else:
                ui = bound0(ci, di)
                vi = bound1(ci, di)
                if vi < ui: ui, vi = vi, ui

                P.append((i, ui))
                P.append((i, vi))

                if cn > ci:
                    if ui == vi:
                        P.pop()
                        S.append(_S(POINT, ui))
                    else:
                        S.append(_S(INTERVAL, ui, vi))
                else:
                    S.append(_S(RAY_UNION, ui, vi))

        P = sorted(P, key=lambda x: x[1])
        for i, p in enumerate(P):
            S[p[0]].map_idx(i, p[1]); P[i] = p[1]

        return P, S

    def __mn0n(self, P, S):
        M_, N0, N_ = self.__m_n0n_(P, S)

        M = [0]
        N = [N0]

        for m_ in M_: M.append(m_ + M[-1])
        M.pop(0)

        for n_ in N_: N.append(n_ + N[-1])
        N.pop(0)

        return M, N0, N

    def __m_n0n_(self, P, S):
        m = len(P)

        M_ = [0 for _ in range(m + 1)]
        N0 = 0
        N_ = [0 for _ in range(m)]

        for s in S:
            if s.type == POINT:
                M_[s.ui_idx]     += 1
                M_[s.ui_idx + 1] -= 1

            if s.type == INTERVAL:
                N_[s.ui_idx] += 1
                N_[s.vi_idx] -= 1

                M_[s.ui_idx] += 1
                M_[s.vi_idx + 1] -= 1

            if s.type == RAY_NEG:
                N0 += 1
                N_[s.ui_idx] -= 1

                M_[0] += 1
                M_[s.ui_idx + 1] -= 1

            if s.type == RAY:
                N_[s.ui_idx] += 1
                M_[s.ui_idx] += 1

            if s.type == RAY_UNION:
                N0 += 1
                N_[s.ui_idx] -= 1
                N_[s.vi_idx] += 1

                M_[0] += 1
                M_[s.vi_idx] += 1
                M_[s.ui_idx + 1] -= 1

            if s.type == R:
                N0 += 1
                M_[0] += 1

        # last element of M_ just for not raising
        # IndexErrors
        M_.pop()

        return M_, N0, N_

    def __intervals(self, M, N0, N, P, n):
        m = len(P)

        res = {k: None for k in self.epsilons}

        for e in self.epsilons:
            intervals = []
            for i in range(m):
                if i == 0 and N0 / n > e:
                    intervals.append([-inf, P[i]])
                elif i == m - 1 and N[i] / n > e:
                    intervals.append([P[i], inf])
                elif N[i] / n > e:
                    intervals.append([P[i], P[i + 1]])
                elif M[i] / n > e:
                    intervals.append([P[i], P[i]])

            if self.convex_hull:
                res[e] = \
                    [[intervals[0][0],intervals[-1][1]]]
            else:
                i = 0; j = 1; idx_reduced = []
                while j < len(intervals):
                    if intervals[i][1] == intervals[j][0]:
                        intervals[i][1] = intervals[j][1]
                    else:
                        idx_reduced.append(i)
                        i = j
                    j += 1
                idx_reduced.append(i)

                res[e] = \
                    [intervals[i] for i in idx_reduced]

        return res

class _S:
    def __init__(self, type, ui = None, vi = None):
        self.type = type
        self.ui     = ui
        self.vi     = vi
        self.ui_idx = None
        self.vi_idx = None

    def map_idx(self, idx, val):
        if self.type in TWO_VALS:
            if val == self.ui:
                self.ui_idx = idx
            else:
                self.vi_idx = idx
        else:
            self.ui_idx = idx

    def __repr__(self):
        if self.type == INTERVAL:
            return "[{}, {}]".format(self.ui, self.vi)
        if self.type == RAY_UNION:
            return "(-inf, {}] & [{}, inf)".format(
                self.ui, self.vi
            )
        if self.type == RAY_NEG:
            return "(-inf, {}]".format(self.ui)
        if self.type == RAY:
            return "[{}, inf)".format(self.ui)
        if self.type == POINT:
            return "[{}]".format(self.ui)
        if self.type == R:
            return "R"
        else:
            return "EMTPY"
