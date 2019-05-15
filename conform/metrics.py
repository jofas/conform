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
