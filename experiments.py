from conform import CP, ICP
from conform.ncs import NC1NN, NCSNeuralNet, \
    NCSDecisionTree, NCSKNearestNeighbors

import h5py
import numpy as np

def load_usps():
    with h5py.File('usps.h5', 'r') as hf:
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
    model.add(Dense( units=out_dim
                   , activation='softmax'))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def mondrian_each_label_nn(x, y): return np.argmax(y)

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
    labels = np.arange(10)

    # icp
    ncs = NCSKNearestNeighbors(labels, n_neighbors=1)

    icp = ICP(ncs, epsilons, labels)
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

    # cp offline
    X_train = np.vstack((X_train, X_cal))
    y_train = np.append(y_train, y_cal)

    ncs = NCSKNearestNeighbors(labels, n_neighbors=1)

    cp = CP(ncs, epsilons, labels)
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)

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
    labels = np.arange(10)

    model = compile_model(X.shape[1], y.shape[1])

    train = lambda X, y: model.fit(X, y, epochs=5, verbose=0)
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
    icp = ICP(ncs, epsilons, labels)
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
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

    # cp offline
    X_train = np.vstack((X_train, X_cal))
    y_train = np.vstack((y_train, y_cal))

    model = compile_model(X.shape[1], y.shape[1])

    cp  = CP(ncs, epsilons, labels)
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)

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
    labels = np.arange(10)

    # icp
    ncs = NCSDecisionTree(min_samples_leaf=50)

    icp = ICP(ncs, epsilons, labels)
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

    # cp offline
    X_train = np.vstack((X_train, X_cal))
    y_train = np.append(y_train, y_cal)

    ncs = NCSDecisionTree(min_samples_leaf=50)

    cp = CP(ncs, epsilons, labels)
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)

def regression():
    from sklearn.linear_model import LinearRegression
    from statistics import mean

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

    X = X.reshape(-1,)

    n = X.shape[0]
    m = mean(X)
    s = sum(y)
    t = sum([(X[i] - m) * y[i] for i in range(n - 1)])
    w = X[n - 1] - m
    u = sum([(X[i] - m) ** 2 for i in range(n)])
    v = sum(X)
    z = v / n

    a = - 1 / n + z * w / u
    b = - w / u

    e = t / u
    f = z * t / u - s / n

    a = round(a,3)
    b = round(b,3)
    e = round(e,3)
    f = round(f,3)

    def c(xi, yi = None):
        if yi == None:
            return 1 + a + xi * b
        else:
            return a + xi * b

    def d(xi, yi = None):
        if yi == None:
            return - xi * e + f
        else:
            return yi - xi * e + f

    # RRCM "kernel method" passes C, D to RRCM

    # RRCM (before just least squares)

    cn = round(c(X[-1]),3)
    dn = round(d(X[-1]),3)
    if cn < 0: cn = -cn; dn = -dn

    bound0 = lambda ci, di: -(di - dn) / (ci - cn)
    bound1 = lambda ci, di: -(di + dn) / (ci + cn)
    bound2 = lambda ci, di: -(di + dn) / 2 * ci

    from infinity import inf

    C = []
    D = []
    P = []

    for (x_, y_) in zip(X, y):
        ci = round(c(x_, y_),3)
        di = round(d(x_, y_),3)
        if ci < 0: ci = -ci; di = -di;

        C.append(ci)
        D.append(di)

        if ci != cn:
            b0 = round(bound0(ci, di),2)
            b1 = round(bound1(ci, di),2)
            P.append(b0)
            P.append(b1)
        elif ci == cn and cn != 0 and di != dn:
            b2 = round(bound2(ci, di),2)
            P.append(b2)

    P = sorted(P)
    P = [-inf] + P + [inf]

    m = len(P) - 1

    N = [0 for _ in range(m)]
    M = [0 for _ in range(m)]

    for i in range(n - 1):
        for j in range(m):
            if  C[i] * P[j] + D[i] >= cn * P[j] + dn:
                M[j] += 1
                if C[i] * P[j+1] + D[i] >= cn * P[j+1] + dn:
                    N[j] += 1

    e = 0.08

    intervals = []
    for j in range(1, m):
        if N[j] / n > e:
            intervals.append([P[j], P[j+1]])
        elif M[j] / n > e:
            intervals.append([P[j], P[j]])

    i = 0; j = 1; idx_reduced = []
    while j < len(intervals):
        if intervals[i][1] == intervals[j][0]:
            intervals[i][1] = intervals[j][1]
        else:
            idx_reduced.append(i)
            i = j
        j += 1

    # Flag for convex hull of intervals
    if len(idx_reduced) > 0:
        reduced = [intervals[i] for i in idx_reduced]
    else:
        reduced = [intervals[0]]
    print(reduced)

def main():
    regression()
    #usps_nc1nn()
    #knn()
    #neural_net()
    #descision_tree()

if __name__ == '__main__':
    main()
