import numpy as np

class CPMetrics:
    def __init__(self, epsilons):
        self.epsilons = epsilons
        self.classes  = {}
        self.sum      = _ClassMetrics(self.epsilons)

    def update(self, predicted, y, k):
        if k not in self.classes:
            self.classes[k] = _ClassMetrics(self.epsilons)
        self.classes[k].update(predicted, y)
        self.sum.update(predicted, y)

    def __repr__(self):
        res = ""
        if len(self.classes) > 1:
            for k in sorted(self.classes.keys()):
                res += "{}  {}".format(
                    "class", self.classes[k].__repr__(7, k)
                )
            res += "{}  {}".format(
                "class", self.sum.__repr__(7, "sum")
            )
        else:
            for k in self.classes:
                res += repr(self.classes[k])
        return res

class _ClassMetrics:
    def __init__(self, epsilons):
        self.eps = {e: _EpsilonMetrics() for e in epsilons}

    def update(self, predicted, y):
        for e in self.eps:
            self.eps[e].update(predicted[e], y)

    def __repr__(self, offset = 0, k = None):
        fill_offset = lambda x: offset * x

        res = "epsilon  status   {}   {}  {}  {}".format(
            "err sum   err mul   err sin",
            "%mul      %sin      %emp",
            "         n      mul      sin      emp",
            "err mul  err sin\n"
        )

        res += "{}{}\n".format(fill_offset("-"), 129 * "-")

        for e in self.eps:
            if k != None:
                res += "{:>5}  ".format(k)

            res += "{:7}  {}  {}\n".format(
                e, self.eps[e].accuracy(e), self.eps[e]
            )

        res += "{}{}\n".format(fill_offset("-"), 129 * "-")

        return res

class _EpsilonMetrics:
    def __init__(self):
        self.n   = 0
        self.mul = 0
        self.sin = 0
        self.emp = 0
        self.err = _Err()

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

    def accuracy(self, e):
        return _EpsilonAccuracy(self, e)

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

class _EpsilonAccuracy:
    def __init__(self, em, e):
        self.prc_mul = em.mul / em.n
        self.prc_sin = em.sin / em.n
        self.prc_emp = em.emp / em.n

        self.err_sum = em.err.sum() / em.n

        self.err_mul = \
            em.err.mul / em.mul if em.mul > 0 else 0.0

        self.err_sin = \
            em.err.sin / em.sin if em.sin > 0 else 0.0

        if round(self.err_sum, 5) <= e:
            self.status = "OK"
        else:
            self.status = "FAILED"

    def __repr__(self):
        return "{:>6}  {}  {}".format(
            self.status,
            "{:> .5f}  {:> .5f}  {:> .5f}".format(
                self.err_sum, self.err_mul, self.err_sin
            ),
            "{:> .5f}  {:> .5f}  {:> .5f}".format(
                self.prc_mul, self.prc_sin, self.prc_emp
            ))

class _Err:
    def __init__(self):
        self.emp = 0
        self.mul = 0
        self.sin = 0

    def sum(self):
        return self.emp + self.mul + self.sin

class RRCMMetrics:
    def __init__(self, epsilons):
        self.eps = {e:_IntervalMetrics() for e in epsilons}

    def update(self, predicted, y):
        for e in predicted:
            self.eps[e].update(predicted[e], y)

    def __repr__(self):
        return str(self.__dict__)

class _IntervalMetrics():
    def __init__(self):
        self.n = 0
        self.ok = 0

    def update(self, predicted, y):
        self.n += 1
        for interval in predicted:
            if interval[0] <= y and y <= interval[1]:
                self.ok += 1
                break

    def accuracy(self, e):
        pass

    def __repr__(self):
        return str(self.__dict__)
