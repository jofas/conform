from conform import CP, ICP
from conform.ncs import NC1NN, NCNeuralNetICP

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
    for i in range(10):
        X, y = load_usps_random()
        epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
        cp = CP(NC1NN(), epsilons, np.arange(10))
        res = cp.score_online(X[:10], y[:10])
        print(i)
        print(res)

def neural_net():
    from keras.models import Sequential
    from keras.layers import Dense

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

    # keras
    model = Sequential()

    model.add(Dense( units=128, activation='tanh'
                   , input_dim=X_train.shape[1] ))
    model.add(Dense(units=128, activation='tanh'))
    model.add(Dense( units=y_train.shape[1]
                   , activation='softmax'))

    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy']
    )

    train = lambda X, y: model.fit(X, y, epochs=5)
    predict = lambda X: model.predict(X)
    icp = ICP(NCNeuralNetICP(train, predict), [0.05], list(range(10)))

    icp.train(X_train, y_train)
    icp.calibrate(X_cal, y_cal)
    res = icp.score(X_test, y_test)
    print(res)

def main():
    neural_net()

if __name__ == '__main__':
    main()
