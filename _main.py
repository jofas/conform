import h5py
import numpy as np

def nn(X, y, x_, y_, d):
    nn_eq       = np.array([])
    nn_eq_dist  = None
    nn_neq      = np.array([])
    nn_neq_dist = None

    for i in range(X.shape[0]):

        dist = d(X[i], x_)

        # ignore same samples
        if dist == 0.0: continue

        # check for label
        if y[i] == y_:
            # check for min
            if nn_eq.shape[0] == 0 or nn_eq_dist > dist:
                nn_eq      = X[i]
                nn_eq_dist = dist
        else:
            # check for min
            if nn_neq.shape[0] == 0 or nn_neq_dist > dist:
                nn_neq      = X[i]
                nn_neq_dist = dist

    return (nn_eq, nn_eq_dist), (nn_neq, nn_neq_dist)

def non_conformity_1nn(X, y, x_, y_):
    d = lambda x, y: np.linalg.norm(x - y)

    nn_eq, nn_neq = nn(X, y, x_, y_, d)

    if nn_eq[1] != None and nn_neq[1] != None:
        return nn_eq[1] / nn_neq[1]
    else:
        return 0.0

class CP:
    def __init__(self, A, epsilon, labels):
        self.A       = A
        self.epsilon = epsilon
        self.labels  = labels
        self.X       = np.array([])
        self.y       = np.array([])

    def update(self, x, y):
        if self.X.shape[0] == 0:
            self.X = np.array([x])
        else:
            self.X = np.vstack((self.X, x))

        self.y = np.append(self.y, y)

    def predict(self, x):

        Gamma = []

        for i in range(len(self.labels)):

            # add the prediction to the data set
            if i > 0:
                self.X = self.X[:-1]
                self.y = self.y[:-1]

            self.update(x, self.labels[i])

            # compute the non-conformity scores
            scores = []
            for j in range(self.X.shape[0]):
                xc = self.X[j]
                yc = self.y[j]
                score = self.A(self.X, self.y, xc, yc)
                scores.append(score)

            # compute p_value
            z_score = scores[-1]
            c = 0
            for score in scores:
                if score >= z_score:
                    c += 1

            p = c / self.X.shape[0]

            if p > self.epsilon:
                Gamma.append(self.labels[i])

        return Gamma


with h5py.File('usps.h5', 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]

    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]


X = np.vstack((X_tr, X_te))
y = np.concatenate((y_tr, y_te))

#indices = np.arange(X.shape[0])
#np.random.shuffle(indices)

#X = X[indices]
#y = y[indices]

labels = np.unique(y)

cp = CP(non_conformity_1nn, 0.05, labels)
Err = 0
Mul = 0
Emp = 0
for i in range(X.shape[0]):
    pred = cp.predict(X[i])
    if y[i] not in pred:
        Err += 1
    if len(pred) > 1:
        Mul += 1
    if len(pred) == 0:
        Emp += 1

    n = i + 1
    print("""n: {}, Err: {}, Mul: {}, Emp: {}, Err %: {}, Mul %: {}, Emp %: {}""".format(
                n, Err, Mul, Emp, Err/n, Mul/n, Emp/n))
    cp.update(X[i], y[i])
