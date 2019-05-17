from conform import CP, ICP
from conform.ncs import NC1NN, NCNeuralNetICP \
                      , NCNeuralNetCP, NCDecisionTreeICP \
                      , NCDecisionTreeCP

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

def usps_nc1nn():
    X, y = load_usps_random()
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]

    cp = CP(NC1NN(), epsilons, np.arange(10))
    res = cp.score_online(X, y)
    print("normal")
    print(res)

    cp = CP(NC1NN(), epsilons, np.arange(10), True)
    res = cp.score_online(X, y)
    print("smoothed")
    print(res)

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

    # icp
    model = compile_model(X.shape[1], y.shape[1])

    train = lambda X, y: model.fit(X, y, epochs=5, verbose=0)
    predict = lambda X: model.predict(X)

    icp = ICP(NCNeuralNetICP(train, predict), [0.05], list(range(10)))
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

    # cp offline
    X_train = np.vstack((X_train, X_cal))
    y_train = np.vstack((y_train, y_cal))

    model = compile_model(X.shape[1], y.shape[1])

    cp = CP(NCNeuralNetCP(train, predict), [0.05], list(range(10)))
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

    # icp
    icp = ICP(NCDecisionTreeICP(min_samples_leaf=50), [0.05], list(range(10)))
    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

    # cp offline
    X_train = np.vstack((X_train, X_cal))
    y_train = np.append(y_train, y_cal)

    cp = CP(NCDecisionTreeCP(min_samples_leaf=50), [0.05], list(range(10)))
    cp.train(X_train, y_train)
    res = cp.score(X_test, y_test)
    print(res)

def main():
    usps_nc1nn()
    #neural_net()
    #descision_tree()

if __name__ == '__main__':
    main()
