from conform import CP
from conform.ncs import NC1NN

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
    from sklearn.neural_network import MLPClassifier

    from keras.models import Sequential
    from keras.layers import Dense

    from keras.wrappers.scikit_learn import KerasClassifier

    X, y = load_usps_random()

    split = int(X.shape[0] / 3)

    y = np.array(
        [[0. if j != v else 1. for j in range(10)] \
            for v in y])

    X_train = X[:2*split]
    y_train = y[:2*split]

    X_test  = X[2*split:]
    y_test  = y[2*split:]

    # keras
    def keras_compile():
        model = Sequential()

        model.add(Dense(
            units=64,
            activation='tanh',
            input_dim=X_train.shape[1]
        ))
        model.add(Dense(units=128, activation='tanh'))
        model.add(Dense( units=y_train.shape[1]
                       , activation='tanh'))

        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    clf = keras_compile()
    #KerasClassifier(keras_compile)
    clf.fit(X_train, y_train)

    print(clf.predict(X_test))
    #res = clf.score(X_test, y_test)
    #print(res)

    #clf = MLPClassifier((128,))
    #clf.fit(X_train, y_train)
    #res = clf.score(X_test, y_test)
    #print(res)

def main():
    neural_net()

if __name__ == '__main__':
    main()
