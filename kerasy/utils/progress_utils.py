# coding: utf-8
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from .vis_utils import measureCanvas

VALUES = {}
INITIAL_TIME = 0

def flush_progress_bar(it, max_iter, metrics={}, barname="", verbose=1, **kwargs):
    """Output the realtime progress.
    @params it      : (int) Current iteration.
    @params max_iter: (int) Maximum number of iterations.
    @params verbose : (int) -1, 0, 1, 2
        -1 = silent
        0  = only progress bar
        1  = progress bar and metrics
        2  = progress plot
    @params metric  : {(str): (float, int)}
    @params barname : (str)
    """
    if verbose<0:
        return
    elif verbose>1:
        flush_progress_plot(it, max_iter, metrics=metrics, **kwargs)
    else:
        global INITIAL_TIME
        if it == 0:
            INITIAL_TIME = time.time()
        if len(barname)>0 and barname[-1]!=" ":
            barname = barname + " "

        it+=1
        digit = len(str(max_iter))
        rate  = it/max_iter
        n_bar = int(rate/0.05)
        bar   = ('#' * n_bar).ljust(20, '-')
        percent = f"{rate*100:.2f}"


        if verbose==1:
            metric = ", ".join([f"{k}: {v}" for  k,v in metrics.items()])
            content = f"\r{barname}{it:>0{digit}}/{max_iter} [{bar}] {percent:>6}% - {time.time()-INITIAL_TIME:.3f}s  {metric}"
        else: # verbose==0
            content = f"\r{barname}{it:>0{digit}}/{max_iter} [{bar}] {percent:>6}% - {time.time()-INITIAL_TIME:.3f}s"
        sys.stdout.write(content)

def flush_progress_plot(it, max_iter, ncols_max=3, metrics={}, **kwargs):
    """If verbose==2, this function will be called.
    """
    num_values=len(metrics)
    ncols, nrows, figsize = measureCanvas(num_values, ncols_max=ncols_max, figinches=(6,4))

    fig,axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(*figsize)

    it+=1
    global VALUES
    for i, (k,v) in enumerate(metrics.items()):
        if it==1:
            VALUES[k] = [v]
        else:
            clear_output(wait=True)
            VALUES[k].append(v)
        if num_values==1:
            axes.plot(VALUES[k])
            axes.set_title(f"{it}/{max_iter} {k}: {v}")
            axes.set_xlabel("iteration")
        else:
            axes[i].plot(VALUES[k])
            axes[i].set_title(f"{it+1}/{max_iter} {k}: {v}")
            axes[i].set_xlabel("iteration")
    if it<max_iter:
        plt.pause(0.05)
        clear_output(wait=True)
    else:
        plt.show()
