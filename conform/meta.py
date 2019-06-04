import numpy as np

from infinity import inf

from sklearn.model_selection import KFold

from . import util

class Meta:
    def __init__( self, M_train, M_predict, B_train
                , B_predict, epsilons, labels ):
        self.M_train   = M_train
        self.M_predict = M_predict
        self.B_train   = B_train
        self.B_predict = B_predict
        self.epsilons  = epsilons
        self.labels    = labels
        self.T         = inf

    def train(self, X, y, k_folds):
        X, y = util.format(X, y)

        kf = KFold(n_splits = k_folds, shuffle = True)

        y_meta = np.array([], dtype=np.int64)
        for idx_train, idx_test in kf.split(X):
            X_train, y_train = X[idx_train], y[idx_train]
            X_test,  y_test  = X[idx_test],  y[idx_test]

            self.B_train(X_train, y_train)

            y_meta = np.append(
                y_meta,
                np.array([1 if p == t else 0 for p, t \
                    in zip(self.B_predict(X_test),y_test)])
            )

        ratios = np.array([])
        y_meta_ = np.array([], dtype=np.int64)
        for idx_train, idx_test in kf.split(X):
            X_train = X[idx_train]
            y_train = y_meta[idx_train]
            X_test  = X[idx_test]
            y_test  = y_meta[idx_test]

            self.M_train(X_train, y_train)

            P = self.M_predict(X_test)
            ratios = np.append(
                ratios,
                np.array([p[1] / p[0] for p in P])
            )
            y_meta_ = np.append(y_meta_, y_test)

        for r, m_, m in zip(ratios, y_meta_, y_meta):
            print(r, m_, m)

        # get ROC points
        # compute ROCCH
        # for each epsilon
        #   intercept with iso -> T

        # fit M, B with X
        pass

    def predict(self, X):
        # get p_vals from M and compute ratio
        # if ratio > T
        #   return B.predict
        # else
        #   return -1
        pass

    def score(self, X, y):
        # acc + rejection rate for each epsilon
        pass
