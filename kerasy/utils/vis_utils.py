# coding: utf-8
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from .generic_utils import handleKeyError
from .metric_utils import silhouette_samples
from .metric_utils import silhouette_score

def measureCanvas(nfigs, ncols_max=2, figinches=(6,4)):
    """
    @params nfigs    : (int) Total number of figures.
    @params ncols_max: (int) The number of columns.
    @params figinches: (tuple) The figure size for 1 plot.
    """
    if nfigs>=ncols_max:
        ncols=ncols_max
        nrows=(nfigs-1)//ncols_max+1
    else:
        ncols=nfigs
        nrows=1
    w,h = figinches
    figsize=(w*ncols,h*nrows)
    return (ncols, nrows, figsize)

def galleryplot(func, argnames, iterator, ncols=5, figsize=(4,4), sharex="none", sharey="none", **kwargs):

    """Plot many figures as a gallery.

    Parameters
    ----------
    @param func         : function which plots each image. NOTE: they must recieve the Axes object.
    @param argnames     : keys of one or more arguments. These are given to `func`.
    @param iterator     : iterator which has one or more values, or zip object of them.
    @param ncols        : number of images arranged horizontally (x-axis).
    @param figsize      : (Widht, Height). image size of each image.
    @param sharex,sharey: bool or {'none', 'all', 'row', 'col'}
    @return fig         : class `matplotlib.figure.Figure` object.
    @return axes        : Axes object or array of Axes objects.

    Examples
    --------
    [1-argument]
    >>> def sample(ax, color):
            ax.scatter(0,0,color=color,s=500)
            return ax
    >>> colors=["red","blue","green","black"]
    >>> fig, axes = galleryplot(func=sample, argnames="color", iterator=colors, ncols=4)
    >>> plt.tight_layout()
    >>> plt.show()

    [multi-args]
    >>> def sample(ax, color, marker):
        ax.scatter(0,0,color=color,s=500, marker=marker)
        return ax
    >>> colors  = ["red","blue","green","black"]
    >>> markers = ["o","s","x","^"]
    >>> fig, axes = galleryplot(func=sample2, argnames=["color","marker"], iterator=zip(colors,markers), ncols=4)
    >>> plt.tight_layout()
    >>> plt.show()
    """
    iter_for_count = copy.deepcopy(iterator)
    nfigs = len([e for e in iter_for_count])
    ncols, nrows, figsize = measureCanvas(nfigs, ncols_max=ncols, figinches=figsize)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=figsize)
    axes = axes if axes.ndim>1 else axes.reshape(1,-1)
    for i,vals in enumerate(iterator):
        kwargs = {argnames:vals} if isinstance(argnames, str) else dict(zip(argnames,vals))
        ax = func(ax=axes[i//ncols, i%ncols], **kwargs)

    return fig, axes

def plot_2D(*data, ax=None, figsize=(4,4), **plotargs):
    """ visualize 2-dimensional data. """
    if ax is None: fig, ax = plt.subplots(figsize=figsize)

    if len(data)==1:
        x = data[0]
        y = np.zeros(shape=len(x))
        label = False
    elif len(data)==2:
        x,y = data
        label = True
    else:
        raise ValueError(f"When passing data, it must contain 1 (only data) or 2 (data, class_label) items. However, it contains {len(data)} items.")
    n_cls = len(np.unique(y))
    for cls in np.unique(y):
        ax.scatter(x[y==cls, 0], x[y==cls, 1], label=f"cls.{cls}" if label else None, color=cm.jet(cls/n_cls))
    ax.set_xlabel("$x$", fontsize=14), ax.set_ylabel("$y$", fontsize=14)
    return ax

def objVSexp(obj, *exp, **options):
    """ Visualize 'Objective variable' vs. 'Explanatory variable' plot.
    @params
    @params var_type: str, 'continuous' or 'discrete' (default='continuous')
    """
    n_features = len(exp)
    if n_features==0 or n_features>3:
        raise ValueError(f"The number of explanatory variables should be 1 or 2, but {n_features}")

    var_type = options.pop("var_type", "continuous")
    figsize  = options.pop("figsize", None)
    ax       = options.pop("ax", None)
    color    = options.pop("color", "#722f37")
    alpha    = options.pop("alpha", .2)
    handleKeyError(["continuous", "discrete"], var_type=var_type)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        projection = None if n_features==1 else "3d"
        ax = fig.add_subplot(1,1,1,projection=projection)

    if n_features==1:
        if var_type=="continuous":
            ax.scatter(*exp, obj, color=color, alpha=alpha)
        else:
            sns.boxplot(*exp, obj, ax=ax)
    else:
        ax.scatter(*exp, obj, color=color, alpha=alpha)

    return ax

def silhouette_plot(X, labels, centers, axes=None, set_style=True):
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
    else:
        ax1,ax2=axes

    ax1 = _left_silhouette_plot(X, labels, ax=ax1, set_style=set_style)
    ax2 = _right_silhouette_plot(X, labels, centers, ax=ax2, set_style=set_style)
    if axes is None:
        plt.suptitle((f"Silhouette analysis"), fontsize=14, fontweight='bold')
        plt.show()
    else:
        return (ax1,ax2)

def _left_silhouette_plot(X, labels, ax=None, set_style=True):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(9, 7)

    silhouette_values = silhouette_samples(X, labels)
    silhouette_avg = np.mean(silhouette_values)

    y_lower = 10
    uniq_labels = np.unique(labels)
    n_clusters = len(uniq_labels)
    for i,k in enumerate(uniq_labels):
        Si_k = np.sort(silhouette_values[labels==k])
        Nk = Si_k.shape[0]
        y_upper = y_lower + Nk

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, Si_k,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * Nk, str(i))
        y_lower = y_upper + 10

    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    if set_style:
        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    return ax

def _right_silhouette_plot(X, labels, centers, ax=None, set_style=True):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(9, 7)

    colors = cm.nipy_spectral(labels.astype(float) / (np.max(labels)+1.))

    ax.scatter(X[:,0], X[:,1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
    ax.scatter(centers[:,0], centers[:,1], marker='o', s=200, alpha=1, c="white", edgecolor='k')
    for i,c in enumerate(centers):
        ax.scatter(c[0], c[1], marker='$%d$' % i, s=50, alpha=1, edgecolor='k')

    if set_style:
        ax.set_title("The visualization of the clustered data.")
        ax.set_xlabel("Feature space for the 1st feature")
        ax.set_ylabel("Feature space for the 2nd feature")
    return ax
