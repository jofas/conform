import numpy as np
import matplotlib.pyplot as plt

from infinity import inf
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
from shapely.geometry import LineString, Point

from sklearn.model_selection import KFold

from .metrics import MetaMetrics
from . import util

class Meta:
    def __init__( self, M_train, M_predict, B_train
                , B_predict, epsilons ):
        self.M_train         = M_train
        self.M_predict       = M_predict
        self.B_train         = B_train
        self.B_predict       = B_predict
        self.epsilons        = epsilons
        self.epsilons_invert = [1 - e for e in epsilons]
        self.Ts              = [inf for _ in epsilons]

    def train(self, X, y, k_folds, plot = False):
        X, y = util.format(X, y)

        kf = KFold(n_splits = k_folds, shuffle = True)

        y_meta_ = np.array([], dtype=np.int64)
        for idx_train, idx_test in kf.split(X):
            X_train, y_train = X[idx_train], y[idx_train]
            X_test,  y_test  = X[idx_test],  y[idx_test]

            self.B_train(X_train, y_train)

            y_meta_ = np.append(
                y_meta_,
                np.array([1 if p == t else 0 for p, t \
                    in zip(self.B_predict(X_test),y_test)])
            )

        self.B_train(X, y)

        ratios = np.array([])
        y_meta = np.array([], dtype=np.int64)
        for idx_train, idx_test in kf.split(X):
            X_train = X[idx_train]
            y_train = y_meta_[idx_train]
            X_test  = X[idx_test]
            y_test  = y_meta_[idx_test]

            self.M_train(X_train, y_train)

            P = self.M_predict(X_test)
            ratios_ = [p[1] / p[0] for p in P]
            ratios  = np.append(ratios, np.array(ratios_))
            y_meta  = np.append(y_meta, y_test)

        self.M_train(X, y_meta)

        S = sorted(zip(ratios, y_meta), reverse=True)

        t = sum(y_meta)
        above = 0

        ROCPTS = []
        for i, s in enumerate(S):
            TP = above
            FP = i - above
            FN = t - above
            TN = len(S) - FN - i

            tpr_denom = TP + FN
            fpr_denom = FP + TN

            ROCPTS.append([TP / tpr_denom, FP / fpr_denom])

            above += s[1]

        ROCPTS = np.array(ROCPTS)

        hull = ConvexHull(ROCPTS)
        vs = []
        for i in hull.vertices:
            vs.append(i)
            if sum(ROCPTS[i]) == 0.0: break
        HULLPTS = ROCPTS[vs]

        iso = lambda target, fpr: target / (1.0 - target) \
                                * (len(S) - t) / t * fpr


        for ti, e in enumerate(self.epsilons_invert):
            iso_line = LineString([
                (0.0, iso(e, 0.0)),
                (1.0, iso(e, 1.0))
            ])

            for i in range(len(HULLPTS) - 1):
                line = LineString([
                    tuple(HULLPTS[i]),
                    tuple(HULLPTS[i + 1])
                ])


                inter = iso_line.intersection(line)

                if type(inter) is Point:
                    inter = np.array(inter.coords)[0]

                    d_inter = euclidean(inter, HULLPTS[i])
                    d_hull = euclidean(
                        HULLPTS[i], HULLPTS[i + 1]
                    )

                    prc = d_inter / d_hull

                    j = hull.vertices[i]
                    j_ = hull.vertices[i + 1]

                    d_ratio = euclidean(
                        ratios[j], ratios[j_]
                    )

                    T = ratios[j] + prc * d_ratio

                    self.Ts[ti] = T
                    break

            if plot:
                iso_line = np.array(list(iso_line.coords))
                plt.plot( iso_line[:,0], iso_line[:,1]
                        , color = "g" )
        if plot:
            plt.scatter(ROCPTS[:,0], ROCPTS[:,1])
            plt.plot(HULLPTS[:,0], HULLPTS[:,1], color="r")
            plt.show()

    def predict(self, X):
        X = util.format(X)

        p_vals = self.M_predict(X)
        pred   = self.B_predict(X)

        res = []
        for pv, pr in zip(p_vals, pred):
            predicted = {}
            ratio = pv[1] / pv[0]

            for i, e in enumerate(self.epsilons):
                if self.Ts[i] < ratio:
                    predicted[e] = pr
                else:
                    predicted[e] = -1

            res.append(predicted)

        return res

    def score(self, X, y):
        X, y = util.format(X, y)

        res = MetaMetrics(self.epsilons)
        predicted = self.predict(X)

        for p_, y_ in zip(predicted, y):
            res.update(p_, y_)

        return res
