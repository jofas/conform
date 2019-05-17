import numpy as np

from .base import CPBase
from .metrics import CPMetrics
from .ncs.base import CPBaseNCS

class CP(CPBase):
    def __init__( self, A, epsilons, labels
                , smoothed = False ):
        if CPBaseNCS not in type(A).__bases__:
            raise Exception("Non-conformity score invalid")
        super().__init__(A, epsilons, labels, smoothed)

    def score_online(self, X, y):
        X, y = super().format(X, y)

        res = CPMetrics(self.epsilons)

        for i in range(X.shape[0]):
            predicted = self.predict(X[i])[0]
            res.update(predicted, y[i])
            self.train(X[i], y[i])

        return res
