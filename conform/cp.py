from .base import CPBase
from .metrics import CPMetrics
from .ncs.base import CPBaseNCS

class CP(CPBase):
    def __init__(self, A, epsilons, labels):
        if CPBaseNCS not in type(A).__bases__:
            raise Exception("Non-conformity score invalid")

        self.A        = A
        self.epsilons = epsilons
        self.labels   = labels

    def train(self, X, y):
        X, y = super().format(X, y)
        self.A.train(X, y)

    def predict(self, X):
        X = super().format(X)
        res = [ None for _ in range(X.shape[0]) ]

        for i in range(X.shape[0]):
            predicted = { e: [] for e in self.epsilons }

            for label in self.labels:
                scores = self.A.scores(X[i], label)
                p = super().p_val(scores)

                for epsilon in self.epsilons:
                    if p > epsilon:
                        predicted[epsilon].append(label)

            res[i] = predicted

        return res

    def score(self, X, y):
        X, y = super().format(X, y)

        res = CPMetrics(self.epsilons)
        predicted = self.predict(X)

        for i in range(X.shape[0]):
            res.update(predicted[i], y[i])

        return res

    def score_online(self, X, y):
        X, y = super().format(X, y)

        res = CPMetrics(self.epsilons)

        for i in range(X.shape[0]):
            predicted = self.predict(X[i])[0]
            res.update(predicted, y[i])
            self.train(X[i], y[i])

        return res
