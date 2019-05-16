import numpy as np

class CPMetrics:
    def __init__(self, epsilons):
        self.eps = {e: EpsilonMetrics() for e in epsilons}

    def update(self, predicted, y):
        for e in self.eps:
            self.eps[e].update(predicted[e], y)

    def accuracy(self):
        return {e: self.eps[e].accuracy(e) \
            for e in self.eps}

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

        y = self.__argmax(y)

        if len(predicted) > 1:
            self.mul += 1
            if y not in predicted: self.err["mul"] += 1

        if len(predicted) == 1:
            self.sin += 1
            if y != predicted[0]: self.err["sin"] += 1

        if len(predicted) == 0:
            self.emp += 1
            self.err["emp"] += 1

    def accuracy(self, eps):
        err = sum(self.err.values())
        return { "status"  : "OK" if err / self.n <= eps \
                    else "FAILED"
               , "% mul"   : self.mul / self.n
               , "% sin"   : self.sin / self.n
               , "% emp"   : self.emp / self.n
               , "err ges" : err / self.n
               , "err mul" : self.err["mul"] / self.mul \
                    if self.mul > 0 else 0.0
               , "err sin" : self.err["sin"] / self.sin \
                    if self.sin > 0 else 0.0 }

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

    def __argmax(self, y):
        if type(y) is np.ndarray:
            return np.argmax(y)
        return y
