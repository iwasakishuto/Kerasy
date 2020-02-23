#coding: utf-8
import sys
from collections import deque
from .progress_utils import flush_progress_bar

class BaseMonitor():
    """ Monitors and reports parameters. """
    def __init__(self, n_target=1, max_iter=300, verbose=1, maxlen=2):
        self.iter = 0
        self.max_iter = max_iter
        self.histories = [deque(maxlen=maxlen) for _ in range(n_target)]
        self.verbose = verbose

    def __repr__(self):
        return f"{super().__repr__()}\n" + \
                "\n".join([f"  - {var} = {val}" for var,val in sorted(self.__dict__.items())])

    def _reset(self):
        self.iter = 0
        for hist in self.histories:
            hist.clear()

    def report(self, *targets, metrics={}, barname="", **kwargs):
        flush_progress_bar(self.iter, self.max_iter, metrics=metrics, barname=barname, verbose=self.verbose, **kwargs)
        for i,target in enumerate(targets):
            self.histories[i].append(target)
        self.iter += 1

class ConvergenceMonitor(BaseMonitor):
    def __init__(self, *tols, max_iter=300, verbose=1):
        self.tols = tols
        super().__init__(max_iter=max_iter, n_target=len(tols), verbose=verbose, maxlen=2)

    @property
    def converged(self):
        return self.iter == self.max_iter or \
               (len(self.histories[0])==2 and all([prev-current < tol for (current, prev), tol in zip(self.histories, self.tols)]))


class ThresholdMonitor(BaseMonitor):
    def __init__(self, *thresholds, max_iter=300, verbose=1):
        """
        @params threshold : (thre, mode) mode should be chosed by 'u'(upper), or 'l'(lower)
        """
        self.thresholds = thresholds
        super().__init__(max_iter=max_iter, n_target=len(thresholds), verbose=verbose, maxlen=1)

    @property
    def converged(self):
        return self.iter == self.max_iter or \
               all([val <= tol if mode[0]=='u' else val >= tol for val, (thre, mode) in zip(self.histories, self.thresholds)])
