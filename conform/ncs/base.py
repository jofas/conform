class CPBaseNCS:
    def train(self, X, y):
        pass

    def scores(self, x, labels):
        pass

class ICPBaseNCS(CPBaseNCS):
    def calibrate(self, X, y):
        pass
