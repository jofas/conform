from sklearn.tree import DecisionTreeClassifier as DTC

from .base import NCSBase

class NCSDecisionTree(NCSBase):
    def __init__(self, **sklearn):
        self.clf = DTC(**sklearn)

    def train(self, X, y):
        self.clf.fit(X, y)

    def scores(self, X, y, _cp):
        nodes = self.clf.apply(X)
        res = []
        for (n_, y_) in zip(nodes, y):
            val = self.clf.tree_.value[n_][0]
            g = self.clf.tree_.n_node_samples[n_]
            try:
                score = 1 - val[y_] / g
                res.append(score)
            except:
                res.append(0.0)
        return res

    def score(self, x, labels):
        try:
            node = self.clf.apply(x.reshape(1, -1))[0]
            val = self.clf.tree_.value[node][0]
            g = self.clf.tree_.n_node_samples[node]
            return [1 - val[l] / g for l in labels]
        except:
            return [0.0 for l in labels]
