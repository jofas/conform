from conform import CP, ICP, RRCM
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
    from conform.ncs import NCSKNearestNeighborsRegressor

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

def main():
    regression()
    #usps_nc1nn()
    #knn()
    #neural_net()
    #descision_tree()

if __name__ == '__main__':
    main()
