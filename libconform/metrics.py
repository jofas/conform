import numpy as np

from infinity import inf
from statistics import mean, median

from . import util

class AbstainMetrics:
    def __init__(self, epsilons):
        self.epsilons = epsilons
        self.classes  = {}
        self.sum      = _ClassAbstainMetrics(self.epsilons)

    def update(self, predicted, epsilon, y, k):
        if k not in self.classes:
            self.classes[k] = \
                _ClassAbstainMetrics(self.epsilons)
        self.classes[k].update(predicted, epsilon, y)
        self.sum.update(predicted, epsilon, y)

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

class _ClassAbstainMetrics:
    def __init__(self, epsilons):
        self.eps = {e: _EpsilonAbstainMetrics(e) \
            for e in epsilons}

    def update(self, predicted, epsilon, y):
        for e in self.eps:
            self.eps[e].update(predicted, epsilon, y)

    def __repr__(self, offset = 0, k = None):
        fill_offset = lambda x: offset * x

        res = "epsilon  status  {}  {}\n".format(
            " %err      %rejected",
            "      n      err  rejected"
        )

        res += "{}{}\n".format(fill_offset("-"), 65 * "-")

        for e in self.eps:
            if k != None:
                res += "{:>5}  ".format(k)

            res += "{:7}  {}  {}\n".format(
                e, self.eps[e].accuracy(e), self.eps[e]
            )

        res += "{}{}\n".format(fill_offset("-"), 65 * "-")

        return res

class _EpsilonAbstainMetrics:
    def __init__(self, true_epsilon):
        self.true_epsilon = true_epsilon
        self.n = 0
        self.err = 0
        self.rej = 0

    def update(self, predicted, epsilon, y):
        self.n += 1
        if self.true_epsilon >= epsilon:
            if predicted != y:
                self.err += 1
        else:
            self.rej += 1

    def accuracy(self, e):
        return _EpsilonAbstainAccuracy(self, e)

    def __repr__(self):
        return "{:7d}  {:7d}  {:8d}".format(
            self.n, self.err, self.rej
        )

class _EpsilonAbstainAccuracy:
    def __init__(self, eam, e):
        if eam.n - eam.rej == 0.0:
            self.prc_err = 0.0
        else:
            self.prc_err = eam.err / (eam.n - eam.rej)
        self.prc_rej = eam.rej / eam.n

        if round(self.prc_err, 5) <= e:
            self.status = "OK"
        else:
            self.status = "FAILED"

    def __repr__(self):
        return "{:>6}  {:> .5f}  {:> .7f}".format(
            self.status, self.prc_err, self.prc_rej
        )

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

        y = util.as_argmax(y)

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
        res = "epsilon  status  err  {}  {}\n".format(
            "    mean width  median width",
            "{:>7}  {:>7}".format("n",  "ok")
        )

        res += "-" * 68 + "\n"

        for e in self.eps:
            res += "{:7}  {}  {}\n".format(
                e, self.eps[e].accuracy(e), self.eps[e]
            )

        res += "-" * 68 + "\n"

        return res

class _IntervalMetrics:
    def __init__(self):
        self.n               = 0
        self.ok              = 0
        self.interval_widths = []

    def update(self, predicted, y):
        self.n += 1
        for interval in predicted:
            if interval[0] != -inf and interval[1] != inf:
                self.interval_widths.append(
                    interval[1] - interval[0]
                )

            if interval[0] <= y and y <= interval[1]:
                self.ok += 1

    def accuracy(self, e):
        return _IntervalAccuracy(self, e)

    def __repr__(self):
        return "{:7d}  {:7d}".format(self.n, self.ok)

class _IntervalAccuracy:
    def __init__(self, im, e):
        self.err = 1 - im.ok / im.n
        self.mean_width = mean(im.interval_widths)
        self.median_width = median(im.interval_widths)

        if round(self.err, 5) <= e:
            self.status = "OK"
        else:
            self.status = "FAILED"

    def __repr__(self):
        return "{:>6}  {:>.5f}  {:>10.2f}  {:>12.2f}" \
            .format(
                self.status, self.err,
                self.mean_width, self.median_width
            )

class VennMetrics:
    def __init__(self):
        self.n      = 0
        self.err    = 0
        self.bounds = [0.0, 0.0]

    def update(self, predicted, y, bounds):
        self.n         += 1
        self.err       += predicted != y
        self.bounds[0] += bounds[0]
        self.bounds[1] += bounds[1]

    def accuracy(self):
        return _VennAccuracy(self)

    def __repr__(self):
        res = "{}  {}\n".format(
            "status  %err     mean lower  mean upper",
            "{:>7}  {:>7}".format("n", "err")
        )

        res += "{}\n".format("-" * 57)

        res += "{}  {:7d}  {:7d}\n".format(
            self.accuracy(), self.n, self.err
        )

        res += "{}\n".format("-" * 57)

        return res

class _VennAccuracy:
    def __init__(self, vm):
        self.err = vm.err / vm.n
        self.mean_bounds = np.array(vm.bounds) / vm.n

        if self.err <= self.mean_bounds[1]:
            self.status = "OK"
        else:
            self.status = "FAILED"

    def __repr__(self):
        return "{:>6}  {:>.5f}  {:>.8f}  {:>.8f}".format(
            self.status, self.err, *self.mean_bounds
        )

class MetaMetrics:
    def __init__(self, epsilons):
        self.eps = {e: _MetaEpsilonMetrics() \
            for e in epsilons}

    def update(self, predicted, y):
        for e in self.eps:
            self.eps[e].update(predicted[e], y)

    def __repr__(self):
        res = "epsilon  status  {}  {}\n".format(
            "    %err  %rejected",
            "      n      err  rejected"
        )

        res += "{}\n".format(64 * "-")

        for e in self.eps:
            res += "{:7}  {}  {}\n".format(
                e, self.eps[e].accuracy(e), self.eps[e]
            )

        res += "{}\n".format(64 * "-")

        return res

class _MetaEpsilonMetrics:
    def __init__(self):
        self.n        = 0
        self.err      = 0
        self.rejected = 0

    def update(self, predicted, y):
        self.n += 1

        if predicted == -1: self.rejected += 1
        elif predicted != y: self.err += 1

    def accuracy(self, e):
        return _MetaEpsilonAccuracy(self, e)

    def __repr__(self):
        return "{:7d}  {:7d}   {:7d}".format(
            self.n, self.err, self.rejected
        )

class _MetaEpsilonAccuracy:
    def __init__(self, mem, e):
        self.prc_err = mem.err / mem.n
        self.prc_rejected = mem.rejected / mem.n

        if round(self.prc_err, 5) <= e:
            self.status = "OK"
        else:
            self.status = "FAILED"

    def __repr__(self):
        return "{:>6}  {:> .5f}   {:> .5f}".format(
            self.status,
            self.prc_err,
            self.prc_rejected
        )
