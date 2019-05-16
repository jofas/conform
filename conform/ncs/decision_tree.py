from sklearn.tree import DecisionTreeClassifier as DTC

from .base import CPBaseNCS, ICPBaseNCS

class __NCDecisionTreeBase:
    def __init__(self, **sklearn):
        self.init = False
        self.X    = None
        self.y    = None

        self.clf = DTC(**sklearn)

        self.scores_ = []

    def train(self, X, y):
        self.__append(X, y)
        self.clf.fit(self.X, self.y)

    def calibrate(self, X, y):
        nodes = self.clf.apply(X)
        for i in range(X.shape[0]):
            val = self.clf.tree_.value[nodes[i]][0]
            g = self.clf.tree_.n_node_samples[nodes[i]]
            self.scores_.append(1 - val[y[i]] / g)

    def scores(self, x, labels):
        node = self.clf.apply(x.reshape(1, -1))[0]
        val = self.clf.tree_.value[node][0]
        g = self.clf.tree_.n_node_samples[node]
        return [self.scores_ + [1 - val[l] / g] \
            for l in labels]

    def __init_data(self, X, y):
        self.X    = X
        self.y    = y
        self.init = True

    def __append(self, X, y):
        if not self.init:
            self.__init_data(X, y)
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.vstack((self.y, y))

class NCDecisionTreeCP(__NCDecisionTreeBase, CPBaseNCS):
    def __init__(self, **sklearn):
        super().__init__(**sklearn)

    def train(self, X, y):
        super().train(X, y)
        self.calibrate(X, y)

class NCDecisionTreeICP(__NCDecisionTreeBase, ICPBaseNCS):
    def __init__(self, **sklearn):
        super().__init__(**sklearn)
