import numpy as np

from infinity import inf

from .metrics import VennMetrics
from .vtx.base import VTXBase
from . import util

INVALID_VTX = "Venn Taxonomy invalid"

class Venn:
    def __init__(self, venn_taxonomy):
        if VTXBase not in type(venn_taxonomy).__bases__:
            raise Exception(INVALID_VTX)

        self.venn_taxonomy = venn_taxonomy
        self.labels        = util.LabelMap()
        self.categories    = {}

        self.init_train = False
        self.X_train    = None
        self.y_train    = None

    def train(self, X, y, override = False):
        X, y = util.format(X, y, self.labels)

        if not override and self.init_train:
            self.X_train, self.y_train = util.append(
                self.X_train, self.y_train, X, y
            )
        else:
            self.X_train, self.y_train = X, y
            self.init_train            = True
            # also reset categories
            self.categories = {}

        cs = self.venn_taxonomy.train(
            self.X_train, self.y_train
        )

        for x, y in zip(self.X_train, self.y_train):
            c = self.venn_taxonomy.category(x, y, True)
            if c not in self.categories:
                self.categories[c] = \
                    _VennCategory(self.labels, y)
            else:
                self.categories[c].update(y)

    def predict(self, X, proba = True):
        X = util.format(X)

        pred, pred_proba = [], []
        for x in X:
            proba_matrix = []

            for y in self.labels:
                c = self.venn_taxonomy.category(x,y,False)

                if c not in self.categories:
                    proba_matrix.append(
                        [0 if label != y else 1 \
                            for label in self.labels]
                    )
                else:
                    proba_matrix.append(
                        self.categories[c].proba(y)
                    )

            proba_matrix = np.array(proba_matrix)

            lower        = -inf
            label        = None
            label_column = None
            for i, column in enumerate(proba_matrix.T):
                q = min(column)
                if q > lower:
                    lower        = q
                    label        = i
                    label_column = column

            pred.append(self.labels.reverse(label))
            if proba:
                upper = max(label_column)
                pred_proba.append(
                    [1.0 - upper, 1.0 - lower]
                )

        if proba:
            return np.array(pred), np.array(pred_proba)
        else:
            return np.array(pred)

    def score(self, X, y):
        X, y = util.format(X, y, self.labels)

        res = VennMetrics()
        predicted = self.predict(X)

        for p_, bounds, y_ in zip(*predicted, y):
            res.update(p_, self.labels.reverse(y_), bounds)

        return res

    def score_online(self, X, y):
        X, y = util.format(X, y, self.labels)

        res = VennMetrics()

        for x_, y_ in zip(X, y):
            p, bounds = self.predict(x_)
            res.update(
                p[0], self.labels.reverse(y_), bounds[0]
            )
            self.train(x_, y_)

        return res

class _VennCategory:
    def __init__(self, labels, y):
        self.distribution  = [0 if label != y else 1 \
            for label in labels]
        self.size          = 1

    def update(self, y):
        self.distribution[y] += 1
        self.size            += 1

    def proba(self, y):
        denom = self.size + 1
        return [d / denom if d != y else (d + 1) / denom \
            for d in self.distribution]
