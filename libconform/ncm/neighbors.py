import numpy as np

from ..base import NCMBase

from sklearn.neighbors import NearestNeighbors

class KNeighborsMean(NCMBase):
    def __init__(self, **sklearn):
        if "n_neighbors" in sklearn:
            sklearn["n_neighbors"] += 1
        else:
            sklearn["n_neighbors"] = 6

        self.clf = NearestNeighbors(**sklearn)
        self.y = []

    def fit(self, X, y):
        self.clf.fit(X)
        self.y = y

    def scores(self, X, y, X_eq_fit):
        if X_eq_fit:
            pass
        else:
            ind = self.clf.kneighbors(X, return_distance=False)
            y_pred = self.y[ind]

            res = []
            for label, row in zip(y, y_pred):
                count_label = [count for label_, count in
                    zip(*np.unique(row, return_counts=True))
                        if label_ == label]
                if len(count_label) == 1:
                    res.append(
                        (len(row) - count_label[0]) / len(row)
                    )
                else:
                    res.append(1.0)
            return np.array(res)
