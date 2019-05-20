from .base import CPBase
from .ncs.base import ICPBaseNCS

class ICP(CPBase):
    def __init__( self, A, epsilons, labels
                , smoothed = False ):
        if ICPBaseNCS not in type(A).__bases__:
            raise Exception("Non-conformity score invalid")

        self.cal_init = False
        self.X_cal    = None
        self.y_cal    = None

        super().__init__(A, epsilons, labels, smoothed)

    def calibrate(self, X, y, append = True):
        X, y = self.format(X, y)
        if append:
            self.X_cal, self.y_cal = self.append(
                self.X_cal, self.y_cal, X, y, self.cal_init
            )
            self.A.calibrate(self.X_cal, self.y_cal)
        else:
            self.A.calibrate(X, y)
