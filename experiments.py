from libconform import CP, ICP, RRCM, Venn, Meta
from libconform.ncs import NC1NN, NCSNeuralNet, \
    NCSDecisionTree, NCSKNearestNeighbors, \
    NCSKNearestNeighborsRegressor
from libconform.vtx import VTXKNearestNeighbors

import h5py
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as NN

def load_usps():
    with h5py.File('data/usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]

        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]


    X = np.vstack((X_tr, X_te))
    y = np.concatenate((y_tr, y_te))

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    return X, y

def load_usps_random():
    X, y = load_usps()

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X, y

def compile_model(in_dim, out_dim):
    from keras.models import Sequential
    from keras.layers import Dense

    # keras
    model = Sequential()

    model.add(Dense( units=128, activation='tanh'
                   , input_dim=in_dim ))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense( units=out_dim
                   , activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def mondrian_each_label_nn(x, y): return np.argmax(y)

def mondrian_each_label(x, y): return y

def usps_nc1nn():
    from time import time
    X, y = load_usps_random()
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]

    start = time()
    cp = CP(NC1NN(), epsilons, np.arange(10))
    res = cp.score_online(X, y)
    print("normal")
    print("time: " + str(time() - start))
    print(res)

    start = time()
    nn = NCSKNearestNeighbors(np.arange(10), n_neighbors=1)
    cp = CP(nn, epsilons, np.arange(10))
    res = cp.score_online(X, y)
    print("new")
    print("time: " + str(time() - start))
    print(res)

    cp = CP(NC1NN(), epsilons, np.arange(10), True)
    res = cp.score_online(X, y)
    print("smoothed")
    print(res)

# knn {{{
def knn():
    X, y = load_usps_random()

    X_cal = X[:200]
    y_cal = y[:200]

    X     = X[200:]
    y     = y[200:]

    split = int(X.shape[0] / 3)

    X_train = X[:2*split]
    y_train = y[:2*split]

    X_test  = X[2*split:]
    y_test  = y[2*split:]

    epsilons = [0.005, 0.01, 0.025, 0.05, 0.1]

    # icp
    ncs = NCSKNearestNeighbors(n_neighbors=1)

    icp = ICP(ncs, epsilons)
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

    # cp offline
    '''
    X_train = np.vstack((X_train, X_cal))
    y_train = np.append(y_train, y_cal)

    ncs = NCSKNearestNeighbors(labels, n_neighbors=1)

    cp = CP(ncs, epsilons)
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)
    '''
# }}}

# neural_net {{{
def neural_net():
    X, y = load_usps_random()
    y = np.array(
        [[0. if j != v else 1. for j in range(10)] \
            for v in y])

    X_cal = X[:200]
    y_cal = y[:200]

    X     = X[200:]
    y     = y[200:]

    split = int(X.shape[0] / 3)

    X_train = X[:2*split]
    y_train = y[:2*split]

    X_test  = X[2*split:]
    y_test  = y[2*split:]

    epsilons = [0.005, 0.01, 0.025, 0.05, 0.1]

    model = compile_model(X.shape[1], y.shape[1])

    train = lambda X, y: model.fit(X, y, epochs=5, verbose=0)
    predict = lambda X: model.predict(X)
    ncs = NCSNeuralNet(train, predict)

    '''
    # icp mondrian
    icp = ICP( ncs, epsilons,
             , mondrian_taxonomy = mondrian_each_label_nn )
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)
    '''

    # icp
    model = compile_model(X.shape[1], y.shape[1])
    icp = ICP(ncs, epsilons)
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

    # only best
    res = icp.predict_best(X_test)
    res = np.array([[res[0][i], res[1][i]] \
        for i in range(X_test.shape[0])])

    print(res)

    '''
    # cp offline mondrian
    X_train = np.vstack((X_train, X_cal))
    y_train = np.vstack((y_train, y_cal))

    model = compile_model(X.shape[1], y.shape[1])

    cp  = CP( ncs, epsilons, labels
            , mondrian_taxonomy = mondrian_each_label_nn )
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)
    '''

    '''
    # cp offline
    X_train = np.vstack((X_train, X_cal))
    y_train = np.vstack((y_train, y_cal))

    model = compile_model(X.shape[1], y.shape[1])

    cp  = CP(ncs, epsilons, labels)
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)
    '''
# }}}

# decision_tree {{{
def descision_tree():
    X, y = load_usps_random()

    X_cal = X[:200]
    y_cal = y[:200]

    X     = X[200:]
    y     = y[200:]

    split = int(X.shape[0] / 3)

    X_train = X[:2*split]
    y_train = y[:2*split]

    X_test  = X[2*split:]
    y_test  = y[2*split:]

    epsilons = [0.005, 0.01, 0.025, 0.05, 0.1]

    # icp
    ncs = NCSDecisionTree(min_samples_leaf=50)

    icp = ICP(ncs, epsilons)
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

    # only best
    res = icp.predict_best(X_test)
    res = np.array([[res[0][i], res[1][i]] \
        for i in range(X_test.shape[0])])

    print(res)
    print(icp.predict_best(X_test, False))

    '''
    # cp offline
    X_train = np.vstack((X_train, X_cal))
    y_train = np.append(y_train, y_cal)

    ncs = NCSDecisionTree(min_samples_leaf=50)

    cp = CP(ncs, epsilons)
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)
    '''
# }}}

# knn_regression {{{
def knn_regression():
    X, y = load_boston(True)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    nn = NCSKNearestNeighborsRegressor(n_neighbors=1)
    clf = RRCM(nn, [0.005, 0.01, 0.025, 0.05, 0.1])
    res = clf.score_online(X, y)
    print("online")
    print(res)

    nn = NCSKNearestNeighborsRegressor(n_neighbors=1)
    clf = RRCM(nn, [0.005, 0.01, 0.025, 0.05, 0.1])
    clf.train(X[:400], y[:400])
    res = clf.score_online(X[400:], y[400:])
    print("400 -> online")
    print(res)

    nn = NCSKNearestNeighborsRegressor(n_neighbors=1)
    clf = RRCM(nn, [0.005, 0.01, 0.025, 0.05, 0.1])
    clf.train(X[:400], y[:400])
    res = clf.score(X[400:], y[400:])
    print("400 -> offline")
    print(res)
# }}}

# venn {{{
def venn():
    X, y = load_usps_random()
    y = np.array(y, dtype=np.float64)

    vtx = VTXKNearestNeighbors(n_neighbors=1)
    clf = Venn(vtx)

    clf.train(X[:8000], y[:8000])
    res = clf.score(X[8000:], y[8000:])
    print(res)

    #clf = Venn(vtx, np.arange(10))
    #res = clf.score_online(X[:500], y[:500])
    #print(res)
# }}}

# oy_venn {{{
def oy_venn():
    from oy.main import meta, standardize, reduce_data
    from oy.import_data import import_data
    from oy.pca import pca

    from sklearn.decomposition import PCA

    X, y, _ = import_data('oy/data/clean.csv')
    print(len(X))
    X, y = reduce_data(X, y, 0.1)

    clf = PCA(n_components = 4)
    X = clf.fit_transform(X)

    print(clf.explained_variance_ratio_)
    print(sum(clf.explained_variance_ratio_))

    X, m = meta(X)
    X = standardize(X, m)

    y = [0 if x == -1.0 else 1 for x in y]
    X, y = np.array(X), np.array(y)

    #print(float(sys.getsizeof(X)) / ( 2**20))

    vtx = VTXKNearestNeighbors(np.arange(2),n_neighbors=3)
    clf = Venn(vtx, np.arange(2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.1)

    clf.train(X_train, y_train)
    res = clf.score(X_test, y_test)
    print(res)
# }}}

# oy_neural_net {{{
def oy_neural_net():
    from oy.main import meta, standardize, reduce_data
    from oy.import_data import import_data
    from oy.pca import pca

    from sklearn.decomposition import PCA

    X, y, _ = import_data('oy/data/clean.csv')
    print(len(X))
    X, y = reduce_data(X, y, 0.1)

    clf = PCA(n_components = 4)
    X = clf.fit_transform(X)

    print(clf.explained_variance_ratio_)
    print(sum(clf.explained_variance_ratio_))

    X, m = meta(X)
    X = standardize(X, m)
    X = np.array(X)

    y = np.array(
        [[0. if j != v else 1. for j in range(10)] \
            for v in y])

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    X_cal = X[:4000]
    y_cal = y[:4000]

    X     = X[4000:]
    y     = y[4000:]

    split = int(X.shape[0] * 0.9)

    X_train = X[:split]
    y_train = y[:split]

    X_test  = X[split:]
    y_test  = y[split:]

    epsilons = [0.005, 0.01, 0.025, 0.05, 0.1]

    model = compile_model(X.shape[1], y.shape[1])

    train = lambda X, y: model.fit(X, y, epochs=5, verbose=1)
    predict = lambda X: model.predict(X)
    ncs = NCSNeuralNet(train, predict)

    '''
    # icp mondrian
    icp = ICP( ncs, epsilons, labels
             , mondrian_taxonomy = mondrian_each_label_nn )
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)
    '''

    # icp
    model = compile_model(X.shape[1], y.shape[1])
    icp = ICP(ncs, epsilons)
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

    # cp
    X_train = np.vstack((X_train, X_cal))
    y_train = np.vstack((y_train, y_cal))
    model = compile_model(X.shape[1], y.shape[1])
    cp = CP(ncs, epsilons)
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)
# }}}

# oy_knn {{{
def oy_knn():
    from oy.main import meta, standardize, reduce_data
    from oy.import_data import import_data
    from oy.pca import pca

    from sklearn.decomposition import PCA

    X, y, _ = import_data('oy/data/clean.csv')
    print(len(X))
    X, y = reduce_data(X, y, 0.01)

    clf = PCA(n_components = 8)
    X = clf.fit_transform(X)

    print(clf.explained_variance_ratio_)
    print(sum(clf.explained_variance_ratio_))

    #X, m = meta(X)
    #X = standardize(X, m)
    y = [0 if x == -1.0 else 1 for x in y]

    X, y = np.array(X), np.array(y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    X_cal = X[:10000]
    y_cal = y[:10000]

    X_train = X[10000:-70000]
    y_train = y[10000:-70000]

    X_test  = X[-70000:]
    y_test  = y[-70000:]

    epsilons = [0.001, 0.005, 0.01, 0.02, 0.025, 0.05]

    ncs = NCSKNearestNeighbors(n_neighbors=1)

    icp = ICP( ncs, epsilons
             , mondrian_taxonomy = mondrian_each_label )
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)
# }}}

# meta {{{
def meta():
    X, y = load_usps_random()

    #X_cal = X[:200]
    #y_cal = y[:200]

    #X     = X[200:]
    #y     = y[200:]

    split = int(X.shape[0] / 10)

    X_train = X[:9*split]
    y_train = y[:9*split]

    X_test  = X[9*split:]
    y_test  = y[9*split:]

    B = NN(n_neighbors=1)

    B_train   = lambda X, y: B.fit(X, y)
    B_predict = lambda X: B.predict(X)

    ncs = NCSKNearestNeighbors(n_neighbors=1)
    M   = ICP(ncs, [])

    def M_train(X, y):
        split = int(X.shape[0] / 5)

        X_train = X[:4*split]
        y_train = y[:4*split]

        X_cal   = X[4*split:]
        y_cal   = y[4*split:]

        M.train(X_train, y_train, override = True)
        M.calibrate(X_cal, y_cal)

    M_predict = lambda X: M.p_vals(X)

    epsilons = [0.01, 0.02, 0.025, 0.03]

    clf = Meta( M_train, M_predict, B_train, B_predict
              , epsilons )

    clf.train(X_train, y_train, k_folds = 10, plot = False)
    res = clf.score(X_test, y_test)
    print(res)
# }}}

# abstain {{{
def abstain():
    X, y = load_usps_random()

    split = int(X.shape[0] / 5)

    X_train = X[:4*split]
    y_train = y[:4*split]

    X_test  = X[4*split:]
    y_test  = y[4*split:]

    epsilons = [0.005, 0.01, 0.025, 0.05, 0.1]

    ncs = NCSKNearestNeighbors(n_neighbors=1)
    cp = CP( ncs, epsilons )
    #       , mondrian_taxonomy = mondrian_each_label )

    cp.train(X_train, y_train)

    res = cp.score(X_test, y_test)
    print(res)

    res = cp.score_abstain(X_test, y_test)
    print(res)

    vtx = VTXKNearestNeighbors(n_neighbors=1)
    clf = Venn(vtx)

    clf.train(X_train, y_train)

    res = clf.score(X_test, y_test)
    print(res)

    res = clf.score_abstain(X_test, y_test, epsilons)
    print(res)
# }}}

from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier as KNN,\
    KNeighborsRegressor as KNNR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.gaussian_process import \
    GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, \
    ConstantKernel as C
from sklearn.svm import SVR

from infinity import inf
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr

plt.style.use("ggplot")

class AbstainPredictor:
    def __init__(
        self, clf_train, clf_predict, clf_score, reward,
        ax = None, e=1e-4, scaled = False
    ):
        self.clf_train   = clf_train
        self.clf_predict = clf_predict
        self.clf_score   = clf_score

        self.reward = reward
        self.T      = 1.0

        self.ax     = ax
        self.e      = e
        self.scaled = scaled

        self.regressors = [
            { "label": "GP [1e-3,1]"
            , "reg": GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1e-3,1.0)))
            , "T": 1.0 },
            { "label": "GP [1e-1,1]"
            , "reg": GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1e-1,1.0)))
            , "T": 1.0 },
            { "label": "GP [1,2]"
            , "reg": GPR(C(1.0, (1e-3,1e3)) * RBF(1.0, (1.0,2.0)))
            , "T": 1.0 },
            { "label": "SVR RBF C100"
            , "reg": SVR(gamma="scale", C=100.0)
            , "T": 1.0 },
            { "label": "SVR RBF C1"
            , "reg": SVR(gamma="scale", C=1.0)
            , "T": 1.0 }
            #{ "label": "Random Forest"
            #, "reg": RFR(n_estimators=10)
            #, "T": 1.0 }
        ]

    def train_cal(self, X, y, ratio=0.1):
        train = int(len(X) * (1 - ratio))
        self.clf_train(X[:train], y[:train])
        self._train(self._A(X[train:], y[train:]))

    def train_kfold(self, X, y, kfolds=5):
        A = []

        kf = KFold(n_splits = kfolds)
        for idx_train, idx_test in kf.split(X):
            X_train, y_train = X[idx_train], y[idx_train]
            X_test,  y_test  = X[idx_test],  y[idx_test]

            self.clf_train(X_train, y_train)

            A += self._A(X_test, y_test)

        self.clf_train(X, y)
        self._train(A)

    def _train(self, A):
        self.T, rew_pts = self._reward_calc(A)
        self._cal_regressors(rew_pts)
        self._plot(rew_pts, "Training set")

    # score {{{
    def score(self, X, y):
        A = self._A(X, y)

        T, rew_pts = self._reward_calc(A)

        rew = self._reward_for_T(self.T, rew_pts)
        print("{:20}  {:.3f}".format("raw T", rew))

        for reg in self.regressors:
            rew = self._reward_for_T(reg["T"], rew_pts)

            print("{:20}  {:.3f}".format(reg["label"],rew))
        print()

        self._plot(rew_pts, "Test set")
    # }}}

    # _reward_calc {{{
    def _reward_calc(self, A):
        A = sorted(A, key=lambda x: x[-1])

        rew,     T       = 0.0, 1.0
        min_rew, max_rew = 0.0, 0.0
        rew_pts          = [[0.0, 0.0]]
        sig_c            = A[0][-1]

        for i, (y, p, sig) in enumerate(A):
            if sig_c != sig:

                rew_pts.append([sig_c, rew])
                if rew > max_rew: max_rew = rew; T = sig_c
                if rew < min_rew: min_rew = rew

                sig_c = sig

            rew += self.reward(p, y)

        rew_pts.append([sig_c, rew])
        if rew > max_rew: max_rew = rew; T = sig_c
        if rew < min_rew: min_rew = rew

        scale_s = lambda x: x / sig_c if self.scaled else x
        scale_r = lambda x: (x - min_rew) \
                          / (max_rew - min_rew)

        rew_pts = np.array([[scale_s(s), scale_r(r)] \
            for s, r in rew_pts])

        return T, rew_pts
    # }}}

    # _cal_regressors {{{
    def _cal_regressors(self, rew_pts):
        X, y = rew_pts[:,0].reshape(-1,1), rew_pts[:,1]

        X_pred = [0.0]
        while X_pred[-1] < X[-1]:
            X_pred.append(X_pred[-1] + self.e)
        X_pred = np.array(X_pred).reshape(-1,1)

        for reg in self.regressors:
            reg["reg"].fit(X,y)

            P = reg["reg"].predict(X_pred)

            max_reward_ = -inf
            for x, p in zip(X_pred, P):
                if p > max_reward_:
                    max_reward_ = p
                    reg["T"]    = x[0]

            pts = np.array([[x, y] for x, y in \
                zip(X_pred[:,0], P)])
            self._plot(pts, reg["label"])
    # }}}

    def _reward_for_T(self, T, Reward_pts):
        res = 0.0
        for sig_lvl_, reward_ in Reward_pts:
            if sig_lvl_ <= T: res = reward_
            else: break
        return res

    def _plot(self, pts, label):
        if self.ax is not None:
            self.ax.plot(pts[:,0], pts[:,1], label=label)
            self.ax.legend()

    def _A(self, X, y):
        return zip(y,self.clf_predict(X),self.clf_score(X))

def bt():
    X, y = load_usps_random()

    #X_train, y_train = X[:-2400],      y[:-2400]
    #X_cal,   y_cal   = X[-2400:-1200], y[-2400:-1200]
    #X_test,  y_test  = X[-1200:],      y[-1200:]

    X_train, y_train = X[:-1200], y[:-1200]
    X_test,  y_test  = X[-1200:], y[-1200:]

    #X_train, y_train = X[:400],    y[:400]
    #X_test,  y_test  = X[400:600], y[400:600]

    reward_fn = lambda pred, true: 1.0 if pred == true \
        else -20.0

    fig, ax = plt.subplots(1,2)

    ax[0].set_title("Conformal Prediction")
    ax[1].set_title("25 Nearest Neighbors")

    ncs = NCSKNearestNeighbors(n_neighbors=1)
    cp = CP(ncs, [], smoothed=False)

    cp_train   = lambda X, y: cp.train(X, y, override=True)
    cp_predict = lambda X: cp.predict_best(X, p_vals=False)
    cp_score   = lambda X: cp.predict_best(X)[1]

    clf = AbstainPredictor(
        cp_train, cp_predict, cp_score, reward_fn,
        ax=ax[0]
    )

    clf.train_kfold(X_train, y_train, kfolds=5)
    #clf.train_cal(X_train, y_train, ratio=0.5)
    clf.score(X_test, y_test)

    knn = KNN(n_neighbors=25)

    knn_train   = lambda X, y: knn.fit(X, y)
    knn_predict = lambda X: knn.predict(X)
    knn_score   = lambda X: \
        [1 - max(row) for row in knn.predict_proba(X)]

    clf = AbstainPredictor(
        knn_train, knn_predict, knn_score, reward_fn,
        ax=ax[1], scaled=True
    )

    clf.train_kfold(X_train, y_train, kfolds=5)
    #clf.train_cal(X_train, y_train, ratio=0.5)
    clf.score(X_test, y_test)

    plt.show()

def main():
    bt()

    #oy_knn()
    #oy_neural_net()
    #oy_venn()

    #venn()
    #knn_regression()
    #usps_nc1nn()
    #meta()

    #knn()
    #neural_net()
    #descision_tree()
    #abstain()

if __name__ == '__main__':
    main()

def regression():

    X = np.array([[5.0],[4.4],[4.9],[4.4],[5.1]
                 ,[5.9],[5.0],[6.4],[6.7],[6.2]
                 ,[5.1],[4.6],[5.0],[5.4],[5.0]
                 ,[6.7],[5.8],[5.5],[5.8],[5.4]
                 ,[5.1],[5.7],[4.6],[4.6],[6.8]])
    y = [ 0.3 , 0.2 , 0.2 , 0.2 , 0.4
        , 1.5 , 0.2 , 1.3 , 1.4 , 1.5
        , 0.2 , 0.2 , 0.6 , 0.4 , 1.0
        , 1.7 , 1.2 , 0.2 , 1.0 , 0.4
        , 0.3 , 1.3 , 0.3 , 0.2 ]

    nn = NCSKNearestNeighborsRegressor(n_neighbors=1)
    clf = RRCM(nn, [0.02, 0.08])
    clf.train(X[:-1], y)
    res = clf.score_online(X[-1], 1.6)
    print(res)

    """
    coeffs_quad, meta_quad = \
        polyfit( Reward_pts_train[:,0]
               , Reward_pts_train[:,1]
               , 2
               , full = True )

    coeffs_cube, meta_cube = \
        polyfit( Reward_pts_train[:,0]
               , Reward_pts_train[:,1]
               , 3
               , full = True )

    quad_f = lambda x: coeffs_quad[0] \
                     + coeffs_quad[1] * x \
                     + coeffs_quad[2] * (x ** 2)

    cube_f = lambda x: coeffs_cube[0] \
                     + coeffs_cube[1] * x \
                     + coeffs_cube[2] * (x ** 2) \
                     + coeffs_cube[3] * (x ** 3)

    x_ = [0.0]
    while x_[-1] < Reward_pts_train[-1,0]:
        x_.append(x_[-1] + 1e-4)

    quad_pts = np.array([[x, quad_f(x)] for x in x_])
    cube_pts = np.array([[x, cube_f(x)] for x in x_])

    ax[0,0].plot(quad_pts[:,0], quad_pts[:,1], c="c")
    ax[0,0].plot(cube_pts[:,0], cube_pts[:,1], c="g")

    """

    """
    kf = KFold(n_splits=5)
    for idx_train, idx_test in kf.split(X):
        X_train, y_train = X[idx_train], y[idx_train]
        X_test,  y_test  = X[idx_test],  y[idx_test]

        cp.train(X_train, y_train, override=True)

        A = zip(y_test, *cp.predict_best(X_test))

        A = sorted(A, key=lambda x: x[-1])

        gain_fn = lambda pred, true: 1.0 if pred == true \
            else 0.0

        loss_fn = lambda pred,true: \
            loss_module.squared(pred, true)

        Gain_pts, Loss_pts, Reward_pts = [], [], []
        gain_pts, loss_pts, reward_pts = [], [], []
        Err_pts, err_pts = [], []

        Gain, Loss, Reward = 0.0, 0.0, 0.0
        Err = 0
        #; Dev = 0.0; Loss = 0.0
        #iso_err = 0.0; iso_loss_ = 0.0
        for i, (y_, p_, sig_lvl_) in enumerate(A):

            if y_ != p_: Err += 1
            err = Err / (i + 1)
            Err_pts.append([sig_lvl_, Err])
            err_pts.append([sig_lvl_, err])

            #if iso_err > err: err = iso_err
            #else: iso_err = err

            #Dev += sig_lvl_ - err
            #dev = Dev / (i + 1)

            #rejected = 1 - ( (i + 1) / len(A) )

            # loss metrics
            #Loss += loss.squared(p_, y_)

            #loss_ = Loss / (i + 1)

            #if iso_loss_ > loss_: loss_ = iso_loss_
            #else: iso_loss_ = loss_

            Gain += gain_fn(p_, y_)
            gain = Gain / (i + 1)
            Gain_pts.append([sig_lvl_, Gain])
            gain_pts.append([sig_lvl_, gain])

            Loss += loss_fn(p_, y_)
            loss = Loss / (i + 1)
            Loss_pts.append([sig_lvl_, Loss])
            loss_pts.append([sig_lvl_, loss])

            Reward = Gain - Loss
            reward = Reward / (i + 1)
            Reward_pts.append([sig_lvl_, Reward])
            reward_pts.append([sig_lvl_, reward])


        gain_pts = np.array(gain_pts)
        loss_pts = np.array(loss_pts)
        reward_pts = np.array(reward_pts)
        err_pts  = np.array(err_pts)

        Gain_pts = np.array(Gain_pts)
        Loss_pts = np.array(Loss_pts)
        Reward_pts = np.array(Reward_pts)
        Err_pts  = np.array(Err_pts)

        from scipy.stats import spearmanr

        corr = spearmanr(Reward_pts[:,1], Err_pts[:,1])
        print(corr)

        # plot loss and acc
        fig, ax = plt.subplots(2,2)

        ax[0,0].plot(gain_pts[:,0], gain_pts[:,1], color="g")
        ax[0,0].plot(loss_pts[:,0], loss_pts[:,1], color="r")
        ax[0,0].plot( reward_pts[:,0], reward_pts[:,1]
                  , color="b" )

        ax[0,0].plot( err_pts[:,0], err_pts[:,1], color="y"
                  , label="err" )
        ax[0,0].plot( err_pts[:,0], err_pts[:,0], color="c"
                  , label="err expected" )

        ax[0,1].scatter(err_pts[:,1], reward_pts[:,1],s=0.1)

        ax[1,0].plot(Gain_pts[:,0], Gain_pts[:,1], color="g")
        ax[1,0].plot(Loss_pts[:,0], Loss_pts[:,1], color="r")
        ax[1,0].plot( Reward_pts[:,0], Reward_pts[:,1]
                  , color="b" )

        ax[1,0].plot( Err_pts[:,0], Err_pts[:,1], color="y"
                  , label="Err" )
        ax[1,0].plot( Err_pts[:,0], Err_pts[:,0], color="c"
                  , label="Err expected" )

        ax[1,1].scatter(Err_pts[:,1], Reward_pts[:,1],s=0.1)

        plt.show()
        break
    """
