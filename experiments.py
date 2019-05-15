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

def main():
    # TODO: usps -> export json? csv?
    for i in range(10):
        X, y = load_usps_random()
        epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
        cp = CP(NC1NN(), epsilons, np.arange(10))
        res = cp.score_online(X[:10], y[:10])
        #print(i)
        #print(res)

if __name__ == '__main__':
    main()
