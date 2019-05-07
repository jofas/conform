import numpy as np

class CP:
    def __init__(self, A, epsilons, labels):
        self.A        = A
        self.epsilons = epsilons
        self.labels   = labels

    def update(self, X, y):
        X, y = self.__format(X, y)
        self.A.update(X, y)

    def predict(self, X):
        X = self.__format(X)
        res = [ None for _ in range(X.shape[0]) ]

        for i in range(X.shape[0]):
            predicted = { e: [] for e in self.epsilons }

            for label in self.labels:
                scores = self.A.scores(X[i], label)
                p = self.__p_val(scores)

                for epsilon in self.epsilons:
                    if p > epsilon:
                        predicted[epsilon].append(label)

            res[i] = predicted

        return res

    def score(self, X, y):
        X, y = self.__format(X, y)

        res = CPMetrics(self.epsilons)
        predicted = self.predict(X)

        for i in range(X.shape[0]):
            res.update(predicted[i], y[i])

        return res

    def score_online(self, X, y):
        X, y = self.__format(X, y)

        res = CPMetrics(self.epsilons)

        for i in range(X.shape[0]):
            predicted = self.predict(X[i])[0]
            res.update(predicted, y[i])
            self.update(X[i], y[i])

        return res

    def __format(self, X, y = None):
        X = self.__list_to_ndarray(X)
        X = self.__reshape_if_vector(X)

        if y is not None:
            y = self.__list_to_ndarray(y)
            y = self.__reshape_if_scalar(y)
            return X, y

        return X

    def __list_to_ndarray(self, z):
        return np.array(z) if type(z) is list else z

    def __reshape_if_vector(self, X):
        return X.reshape(1, X.shape[0]) \
            if len(X.shape) == 1 else X

    def __reshape_if_scalar(self, y):
        return np.array([y]) if type(y) is not np.ndarray \
            else y

    def __p_val(self, scores):
        return sum([1 for s in scores if s >= scores[-1]])\
             / len(scores)

class CPMetrics:
    def __init__(self, epsilons):
        self.eps = {e: EpsilonMetrics() for e in epsilons}

    def update(self, predicted, y):
        for e in self.eps:
            self.eps[e].update(predicted[e], y)

    def __iadd__(self, other):
        for e in self.eps:
            self.eps[e] += other.eps[e]

        return self

    def __repr__(self):
        return str(self.__dict__['eps'])

class EpsilonMetrics:
    def __init__(self):
        self.n   = 0
        self.mul = 0
        self.sin = 0
        self.emp = 0
        self.err = { "emp": 0, "mul": 0, "sin": 0 }

    def update(self, predicted, y):
        self.n += 1

        if len(predicted) > 1:
            self.mul += 1
            if y not in predicted: self.err["mul"] += 1

        if len(predicted) == 1:
            self.sin += 1
            if y != predicted[0]: self.err["sin"] += 1

        if len(predicted) == 0:
            self.emp += 1
            self.err["emp"] += 1

    def __iadd__(self, other):
        for k in self.__dict__:
            if k == 'err':
                for kk in self.err:
                    self.err[kk] += other.err[kk]
            else:
                self.__dict__[k] += other.__dict__[k]

        return self

    def __repr__(self):
        return str(self.__dict__)
