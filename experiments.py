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
from infinity import inf
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

class AbstainPredictor:
    def __init__(self, clf, reward):
        self.clf        = clf
        self.reward     = reward
        self.T          = 1.0
        self.Reward_pts = []

    def train(self, X, y):
        self.clf.train(X, y)

    def calibrate(self, X, y):
        A = zip(y, *self.clf.predict_best(X))
        A = sorted(A, key=lambda x: x[-1])

        Reward     = 0.0
        max_reward = -inf

        prev_sig_lvl_ = inf

        for y_, p_, sig_lvl_ in A:
            if prev_sig_lvl_ == inf:
                prev_sig_lvl_ = sig_lvl_

            elif prev_sig_lvl_ != sig_lvl_:
                self.Reward_pts.append(
                    [prev_sig_lvl_, Reward]
                )
                prev_sig_lvl_ = sig_lvl_

            Reward += self.reward(p_, y_)

            if Reward > max_reward:
                max_reward = Reward
                self.T     = sig_lvl_

        self.Reward_pts.append([prev_sig_lvl_, Reward])

    def score(self, X, y):
        A = zip(y, *self.clf.predict_best(X))
        A = sorted(A, key=lambda x: x[-1])

        Reward = 0.0
        Reward_pts = []

        max_reward, T = -inf, 1.0

        T_reward = 0.0

        prev_sig_lvl_ = inf

        for y_, p_, sig_lvl_ in A:
            if prev_sig_lvl_ == inf:
                prev_sig_lvl_ = sig_lvl_

            elif prev_sig_lvl_ != sig_lvl_:
                Reward_pts.append([prev_sig_lvl_, Reward])
                prev_sig_lvl_ = sig_lvl_

            Reward += self.reward(p_, y_)

            if sig_lvl_ <= self.T: T_reward = Reward

            if Reward > max_reward:
                max_reward = Reward
                T          = sig_lvl_

        Reward_pts.append([prev_sig_lvl_, Reward])

        Reward_pts       = np.array(Reward_pts)
        Reward_pts_train = np.array(self.Reward_pts)

        coeffs_quad = polyfit( Reward_pts_train[:,0]
                             , Reward_pts_train[:,1]
                             , 2 )

        coeffs_cube = polyfit( Reward_pts_train[:,0]
                             , Reward_pts_train[:,1]
                             , 3 )

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

        print(max_reward, T_reward, T_reward / max_reward)
        print(self.T, T)

        fig, ax = plt.subplots(1,1)

        ax.plot( Reward_pts[:,0]
                    , Reward_pts[:,1]
                    , c = "r" )
        ax.plot( Reward_pts_train[:,0]
                    , Reward_pts_train[:,1]
                    , c = "b" )

        ax.plot(quad_pts[:,0], quad_pts[:,1], c="c")
        ax.plot(cube_pts[:,0], cube_pts[:,1], c="g")

        plt.show()

def bt():
    import loss as loss_module

    X, y = load_usps_random()

    X_train, y_train = X[:-2400],      y[:-2400]
    X_cal,   y_cal   = X[-2400:-1200], y[-2400:-1200]
    X_test,  y_test  = X[-1200:],      y[-1200:]

    ncs = NCSKNearestNeighbors(n_neighbors=1)
    cp = CP(ncs, [])

    gain_fn = lambda pred, true: 1.0 if pred == true \
        else 0.0

    loss_fn = lambda pred,true: \
        loss_module.squared(pred, true)

    reward_fn = lambda pred, true: \
        gain_fn(pred, true) - loss_fn(pred, true)

    clf = AbstainPredictor(cp, reward_fn)
    clf.train(X_train, y_train)
    clf.calibrate(X_cal, y_cal)

    clf.score(X_test, y_test)

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

