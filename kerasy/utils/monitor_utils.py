# coding: utf-8
import sys
import time
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output

from .vis_utils import measureCanvas
from .generic_utils import priColor
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
                "\n".join([f"\t- {var} = {val}" for var,val in sorted(self.__dict__.items())])

    def _reset(self):
        self.iter = 0
        for hist in self.histories:
            hist.clear()

    def report(self, *targets, metrics={}, barname="", **kwargs):
        flush_progress_bar(self.iter, self.max_iter, metrics=metrics, barname=barname, verbose=self.verbose, **kwargs)
        for i,target in enumerate(targets):
            self.histories[i].append(target)
        self.iter += 1

class ProgressMonitor():
    """
    Monitor the loop progress.
    @params max_iter: (int) Maximum number of iterations.
    @params verbose : (int) -1, 0, 1, 2
        -1 = silent
        0  = only progress bar
        1  = progress bar and metrics
        2  = progress plot
    @params barname : (str)
    ~~~
    examples)
    >>> from kerasy.utils import ProgressMonitor
    >>> max_iter = 100
    >>> monitor = ProgressMonitor(max_iter=max_iter, verbose=1, barname="NAME")
    >>> for it in range(max_iter):
    >>>     monitor.report(it, loop=it)
    >>> monitor.remove()

    NAME 100/100[####################]100.00% - 0.010[s]  loop: 99
    """
    def __init__(self, max_iter, verbose=1, barname="", **kwargs):
        self._reset()
        self.max_iter = max_iter
        self.digit = len(str(max_iter))
        self.verbose = verbose
        self.barname = barname + " " if len(barname)>0 else ""
        self.ncols_max = kwargs.pop("ncols_max", 3)
        self.report = {
            -1 : self._report_silent,
             0 : self._report_only_prograss_bar,
             1 : self._report_progress_bar_and_metrics,
             2 : self._report_progress_plot,
        }.get(verbose, self._report_progress_bar_and_metrics)
        self.report(it=-1)

    def _reset(self):
        self.histories = {}
        self.iter = 0
        self.initial_seconds_since_epoch = time.time()

    def _report_silent(self, it, **metrics):
        pass

    def _report_only_prograss_bar(self, it, **metrics):
        it += 1
        sys.stdout.write(
            f"\r{self.barname}{it:>0{self.digit}}/{self.max_iter} " + \
            f"[{('#' * int((it/self.max_iter)/0.05)).ljust(20, '-')}]" + \
            f"{it/self.max_iter:>7.2%} - {time.time()-self.initial_seconds_since_epoch:.3f}[s]"
        )

    def _report_progress_bar_and_metrics(self, it, **metrics):
        it += 1
        metric = ", ".join([
            f"{priColor.color(k, color='ACCENT')}: {priColor.color(v, color='BLUE')}" \
            for  k,v in metrics.items()
        ])
        sys.stdout.write(
            f"\r{self.barname}{it:>0{self.digit}}/{self.max_iter}" + \
            f"[{('#' * int((it/self.max_iter)/0.05)).ljust(20, '-')}]" + \
            f"{it/self.max_iter:>7.2%} - {time.time()-self.initial_seconds_since_epoch:.3f}[s]   " + \
            f"{metric}"
        )

    def _report_progress_plot(self, it, **metrics):
        num_values=len(metrics)
        ncols, nrows, figsize = measureCanvas(num_values, ncols_max=self.ncols_max, figinches=(6,4))
        fig,axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.set_size_inches(*figsize)

        if it==-1:
            pass
        elif it==0:
            self.histories.update({k:[v] for k,v in metrics.items()})
        else:
            it+=1
            for i, (k,v) in enumerate(metrics.items()):
                clear_output(wait=True)
                self.histories[k].append(v)
                if num_values==1:
                    axes.plot(self.histories[k])
                    axes.set_title(f"{it}/{self.max_iter} {k}: {v}")
                    axes.set_xlabel("iteration")
                else:
                    axes[i].plot(self.histories[k])
                    axes[i].set_title(f"{it+1}/{self.max_iter} {k}: {v}")
                    axes[i].set_xlabel("iteration")
        plt.pause(0.05)
        clear_output(wait=True)

    def remove(self):
        def _pass():
            pass
        {
            -1: _pass,
             0: print,
             1: print,
             2: plt.show,
        }.get(self.verbose, print)()

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
