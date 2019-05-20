from sklearn.tree import DecisionTreeClassifier as DTC

from .base import CPBaseNCS, ICPBaseNCS

class __NCDecisionTreeBase:
    def __init__(self, **sklearn):
        self.clf = DTC(**sklearn)
        self.scores_ = []

    def train(self, X, y):
        self.clf.fit(X, y)

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

class NCDecisionTreeCP(__NCDecisionTreeBase, CPBaseNCS):
    def __init__(self, **sklearn):
        super().__init__(**sklearn)

    def train(self, X, y):
        super().train(X, y)
        self.calibrate(X, y)

class NCDecisionTreeICP(__NCDecisionTreeBase, ICPBaseNCS):
    def __init__(self, **sklearn):
        super().__init__(**sklearn)
