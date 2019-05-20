import numpy as np

from .base import CPBaseNCS, ICPBaseNCS

class __NCNeuralNetBase:
    def __init__( self, train_, predict_, scorer, gamma):
        self.train_   = train_
        self.predict_ = predict_
        self.scorer   = self.__scorer(scorer)
        self.gamma    = gamma

        self.scores_ = []

    def train(self, X, y):
        self.train_(X, y)

    def calibrate(self, X, y):
        pred = self.predict_(X)
        for i in range(X.shape[0]):
            label = np.argmax(y[i])
            self.scores_.append(self.scorer(pred[i], label))

    def scores(self, x, labels):
        pred = self.predict_(x.reshape(1, -1))
        return [self.scores_ + [self.scorer(pred[0], l)] \
            for l in labels]

    def __scorer(self, scorer):
        if type(scorer) is str:
            if scorer == "sum" : return self.__sum
            if scorer == "diff": return self.__diff
            return self.__max
        else:
            return scorer

    def __max(self, pred, label):
        max_neq = self.__get_max_neq(pred, label)
        return max_neq / (pred[label] + self.gamma)

    def __sum(self, pred, label):
        sum_neq = sum(
            [pred[i] for i in range(pred.shape[0]) \
                if i != label])
        return sum_neq / (pred[label] + self.gamma)

    def __diff(self, pred, label):
        max_neq = self.__get_max_neq(pred, label)
        return max_neq - pred[label]

    def __get_max_neq(self, pred, label):
        return max([pred[i] for i in range(pred.shape[0]) \
            if i != label])

class NCNeuralNetCP(__NCNeuralNetBase, CPBaseNCS):
    def __init__( self, train_, predict_, scorer = "max"
                , gamma = 0.0 ):
        super().__init__(train_, predict_, scorer, gamma)

    def train(self, X, y):
        super().train(X, y)
        self.calibrate(X, y)

class NCNeuralNetICP(__NCNeuralNetBase, ICPBaseNCS):
    def __init__( self, train_, predict_, scorer = "max"
                , gamma = 0.0 ):
        super().__init__(train_, predict_, scorer, gamma)
