class NCSBase:
    def train(self, X, y):
        pass

    def scores(self, X, y, cp):
        pass

    def score(self, x, labels):
        pass

class NCSBaseRegressor:
    def train(self, X, y):
        pass

    def coeffs(self, X, y, cp):
        pass

    def coeffs_n(self, x):
        pass
