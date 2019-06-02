from infinity import inf
from sklearn.neighbors import NearestNeighbors as NN

from .base import VTXBase

class VTXKNearestNeighbors(VTXBase):
    def __init__(self, labels, **sklearn):
        if "n_neighbors" in sklearn:
            sklearn["n_neighbors"] += 1
        else:
            sklearn["n_neighbors"] = 6

        self.labels = labels
        self.nn     = NN(**sklearn)
        self.y      = []
        self.n      = self.nn.n_neighbors
        self.n_cur  = 0

    def train(self, X, y):
        self.nn.fit(X, y)
        self.y = y
        self.n_cur = self.n if len(X) > self.n else len(X)

    def category(self, x, y, contains_x):
        if self.n_cur < 2:
            return y

        x = x.reshape(1,-1)
        nns = self.nn.kneighbors(
            x, n_neighbors = self.n_cur
        )[1][0]

        if contains_x: nns = nns[1:]
        else:          nns = nns[:-1]

        # no find label in y with highest count
        count = { k: 0 for k in self.labels }
        for idx in nns:
            count[self.y[idx]] += 1

        max = -inf
        label = None
        for k, v in count.items():
            if v > max:
                max = v
                label = k

        return label
