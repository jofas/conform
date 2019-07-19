import numpy as np
import random

from .metrics import CPMetrics, AbstainMetrics
from .ncs.base import NCSBase
from . import util

INVALID_A = "Non-conformity classifier invalid"

class _CPBase:
    def __init__( self, A, epsilons, smoothed
                , mondrian_taxonomy ):

        if NCSBase not in type(A).__bases__:
            raise Exception(INVALID_A)

        self.A                 = A
        self.epsilons          = epsilons
        self.labels            = util.LabelMap()
        self.smoothed          = smoothed
        self.mondrian_taxonomy = mondrian_taxonomy

        self.init_train = False
        self.X_train    = None
        self.y_train    = None

        self.nc_scores  = []
        self.ks         = []

    def train(self, X, y, override = False):
        X, y = util.format(X, y, self.labels)

        if not override and self.init_train:
            self.X_train, self.y_train = util.append(
                self.X_train, self.y_train, X, y
            )
        else:
            self.X_train, self.y_train = X, y
            self.init_train            = True

        self.A.train(self.X_train, self.y_train)

    def set_nc_scores(self, X, y, cp):
        self.nc_scores = self.A.scores(X, y, cp)
        self.ks        = [self.mondrian_taxonomy(x_,y_) \
            for (x_,y_) in zip(X,y)]

    def predict(self, X):
        X = util.format(X)

        return np.array([[[1 if p_val > sig_lvl else -1
            for p_val in p_vec]
                for sig_lvl in self.epsilons]
                    for p_vec in self.p_vals(X)])

    def predict_best(self, X, p_vals = True):
        X = util.format(X)

        pred = []; p_vals_ = []
        for p_vec in self.p_vals(X):

            ps = [(l, p) for l, p in zip(
                self.labels, p_vec
            )]

            ps = sorted(ps,key=lambda x: x[1],reverse=True)

            j = 1
            while j < len(self.labels) - 1:
                if ps[0][1] != ps[j][1]: break
                j += 1

            if j == 1:
                pred.append(self.labels.reverse(ps[0][0]))
            else:
                pred.append(random.choice([
                    self.labels.reverse(ps[i][0]) \
                        for i in range(j)
                ]))

            if p_vals:
                p_vals_.append(ps[j][1])

        if p_vals:
            return np.array(pred), np.array(p_vals_)
        else:
            return np.array(pred)

    def score(self, X, y):
        X, y = util.format(X, y, self.labels)

        res = CPMetrics(self.epsilons)
        predicted = self.predict(X)

        for p_, x_, y_ in zip(predicted, X, y):
            k = self.mondrian_taxonomy(x_, y_)
            res.update(p_, self.labels.reverse(y_), k)

        return res

    def score_abstain(self, X, y):
        X, y = util.format(X, y, self.labels)

        res = AbstainMetrics(self.epsilons)
        pred, epsilons = self.predict_best(X)

        for p_, e_, x_, y_ in zip(pred, epsilons, X, y):
            k = self.mondrian_taxonomy(x_, y_)
            res.update(p_, e_, self.labels.reverse(y_), k)

        return res

    def p_vals(self, X):
        X = util.format(X)

        p_fn = self.__p_vals_smoothed if self.smoothed \
            else self.__p_vals

        return np.array([[
            p_fn(s, self.mondrian_taxonomy(x, l)) \
                for s, l in zip(
                    self.A.score(x, self.labels),
                    self.labels
                )
        ] for x in X])

    def __p_vals_smoothed(self, s, k):
        eq, greater, cc = self.__p_val_counter(s, k)
        return (greater + random.uniform(0.0, 1.0) * eq) \
             / cc

    def __p_vals(self, s, k):
        eq, greater, cc = self.__p_val_counter(s, k)
        return (eq + greater) / cc

    def __p_val_counter(self, s, k):
        eq = 0; greater = 0; class_count = 0
        for (k_, s_) in zip(self.ks, self.nc_scores):
            if k_ == k:
                class_count += 1
                if s_ == s: eq += 1
                if s_ >  s: greater += 1
        return eq + 1, greater, class_count + 1

def _not_mcp(x, y): return 0

class CP(_CPBase):
    def __init__( self, A, epsilons, smoothed=False
                , mondrian_taxonomy=_not_mcp ):
        super().__init__( A, epsilons, smoothed
                        , mondrian_taxonomy )

    def train(self, X, y, override = False):
        super().train(X, y, override)
        self.set_nc_scores( self.X_train, self.y_train
                          , cp = True )

    def score_online(self, X, y):
        X, y = util.format(X, y, self.labels)

        res = CPMetrics(self.epsilons)

        for x_, y_ in zip(X, y):
            k = self.mondrian_taxonomy(x_, y_)
            p = self.predict(x_)[0]
            res.update(p, self.labels.reverse(y_), k)
            self.train(x_, y_)

        return res

class ICP(_CPBase):
    def __init__( self, A, epsilons, smoothed=False
                , mondrian_taxonomy=_not_mcp ):
        self.init_cal = False
        self.X_cal    = None
        self.y_cal    = None

        super().__init__( A, epsilons, smoothed
                        , mondrian_taxonomy )

    def calibrate(self, X, y, override = False):
        X, y = util.format(X, y, self.labels)

        if not override and self.init_cal:
            self.X_cal, self.y_cal = util.append(
                self.X_cal, self.y_cal, X, y
            )
        else:
            self.X_cal, self.y_cal = X, y
            self.init_cal          = True

        self.set_nc_scores( self.X_cal, self.y_cal
                          , cp = False )
