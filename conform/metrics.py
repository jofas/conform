import numpy as np

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
        res = "epsilon  status   {}   {}  {}  {}\n".format(
            "err ges   err mul   err sin",
            "%mul      %sin      %emp",
            "         n      mul      sin      emp",
            "err mul  err sin"
        )

        for e in self.eps:
            res += "{:7}  {}  {}\n".format(
                e, self.eps[e].accuracy(e), self.eps[e]
        )

        return res

class EpsilonMetrics:
    def __init__(self):
        self.n   = 0
        self.mul = 0
        self.sin = 0
        self.emp = 0
        self.err = Err()

    def update(self, predicted, y):
        self.n += 1

        y = self.__argmax(y)

        if len(predicted) > 1:
            self.mul += 1
            if y not in predicted: self.err.mul += 1

        if len(predicted) == 1:
            self.sin += 1
            if y != predicted[0]: self.err.sin += 1

        if len(predicted) == 0:
            self.emp += 1; self.err.emp += 1

    def accuracy(self, eps):
        return EpsilonAccuracy(self, eps)

    def __iadd__(self, other):
        for k in self.__dict__:
            if k == 'err':
                for kk in self.err:
                    self.err[kk] += other.err[kk]
            else:
                self.__dict__[k] += other.__dict__[k]

        return self

    def __repr__(self):
        return "{:7d}  {:7d}  {:7d}  {:7d}  {:7d}  {:7d}" \
            .format(
                self.n, self.mul, self.sin, self.emp,
                self.err.mul, self.err.sin
            )

    def __argmax(self, y):
        if type(y) is np.ndarray:
            return np.argmax(y)
        return y

class EpsilonAccuracy:
    def __init__(self, em, eps):
        self.prc_mul = em.mul / em.n
        self.prc_sin = em.sin / em.n
        self.prc_emp = em.emp / em.n

        self.err_ges = em.err.ges() / em.n

        self.err_mul = \
            em.err.mul / em.mul if em.mul > 0 else 0.0

        self.err_sin = \
            em.err.sin / em.sin if em.sin > 0 else 0.0

        if round(self.err_ges, 5) <= eps:
            self.status = "OK"
        else:
            self.status = "FAILED"

    def __repr__(self):
        return "{:>6}  {}  {}".format(
            self.status,
            "{:> .5f}  {:> .5f}  {:> .5f}".format(
                self.err_ges, self.err_mul, self.err_sin
            ),
            "{:> .5f}  {:> .5f}  {:> .5f}".format(
                self.prc_mul, self.prc_sin, self.prc_emp
            ))

class Err:
    def __init__(self):
        self.emp = 0
        self.mul = 0
        self.sin = 0

    def ges(self):
        return self.emp + self.mul + self.sin
