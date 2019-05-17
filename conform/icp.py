from .base import CPBase
from .ncs.base import ICPBaseNCS

class ICP(CPBase):
    def __init__( self, A, epsilons, labels
                , smoothed = False ):
        if ICPBaseNCS not in type(A).__bases__:
            raise Exception("Non-conformity score invalid")
        super().__init__(A, epsilons, labels, smoothed)

    def calibrate(self, X, y):
        X, y = super().format(X, y)
        self.A.calibrate(X, y)
