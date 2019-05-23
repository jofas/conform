import numpy as np
import random

from .metrics import CPMetrics
from .ncs.base import NCSBase
from . import _data_obs as _do

class __CPBase:
    def __init__( self, A, epsilons, labels, smoothed
                , mondrian_taxonomy ):

        if NCSBase not in type(A).__bases__:
            raise Exception("Non-conformity score invalid")

        self.A                 = A
        self.epsilons          = epsilons
        self.labels            = labels
        self.smoothed          = smoothed
        self.mondrian_taxonomy = mondrian_taxonomy

        self.init_train = False
        self.X_train    = None
        self.y_train    = None

        self.nc_scores  = []
        self.ks         = []

    def train(self, X, y, override = False):
        X, y = _do.format(X, y)

        if not override and self.init_train:
            self.X_train, self.y_train = _do.append(
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
        X = _do.format(X)

        res = []
        for x in X:
            predicted = { e: [] for e in self.epsilons }

            score_per_label = self.A.score(x, self.labels)

            for (l,s) in zip(self.labels,score_per_label):
                k = self.mondrian_taxonomy(x, l)
                p = self.__p_val_smoothed(s,k) \
                    if self.smoothed else self.__p_val(s,k)

                for epsilon in self.epsilons:
                    if p > epsilon:
                        predicted[epsilon].append(l)

            res.append(predicted)

        return res

    def score(self, X, y):
        X, y = _do.format(X, y)

        res = CPMetrics(self.epsilons)
        predicted = self.predict(X)

        for (p_, x_, y_) in zip(predicted, X, y):
            k = self.mondrian_taxonomy(x_, y_)
            res.update(p_, y_, k)

        return res

    def __p_val_smoothed(self, s, k):
        eq, greater, cc = self.__p_val_counter(s, k)
        return (greater + random.uniform(0.0, 1.0) * eq) \
             / cc

    def __p_val(self, s, k):
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

class CP(__CPBase):
    def __init__( self, A, epsilons, labels, smoothed=False
                , mondrian_taxonomy=_not_mcp ):
        super().__init__( A, epsilons, labels, smoothed
                        , mondrian_taxonomy )

    def train(self, X, y, override = False):
        super().train(X, y, override)
        self.set_nc_scores( self.X_train, self.y_train
                          , cp = True )

    def score_online(self, X, y):
        X, y = _do.format(X, y)

        res = CPMetrics(self.epsilons)

        count = 0
        for (x_, y_) in zip(X, y):
            count += 1
            if count % 500 == 0: print(str(count) + "\r")
            k = self.mondrian_taxonomy(x_, y_)
            p = self.predict(x_)[0]
            res.update(p, y_, k)
            self.train(x_.reshape(1,-1), y_)

        return res

class ICP(__CPBase):
    def __init__( self, A, epsilons, labels, smoothed=False
                , mondrian_taxonomy=_not_mcp ):
        self.init_cal = False
        self.X_cal    = None
        self.y_cal    = None

        super().__init__( A, epsilons, labels, smoothed
                        , mondrian_taxonomy )

    def calibrate(self, X, y, override = False):
        X, y = _do.format(X, y)

        if not override and self.init_cal:
            self.X_cal, self.y_cal = _do.append(
                self.X_cal, self.y_cal, X, y
            )
        else:
            self.X_cal, self.y_cal = X, y
            self.init_cal          = True

        self.set_nc_scores( self.X_cal, self.y_cal
                          , cp = False )
