import numpy as np

from ..base import ICPBaseNCS

class NCNeuralNetICP(ICPBaseNCS):
    def __init__(self, train, predict):
        self.train   = train
        self.predict = predict
        self.scores  = np.array([])

    def train(self, X, y):
        self.train(X, y)

    def calibrate(self, X, y):
        pred = self.predict(X)
        # do scores and stuff
        pass

    def scores(self, x, labels):
        pred = self.predict(x)
        # do scores stuff
        pass
