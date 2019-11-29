# coding: utf-8
import copy
import matplotlib.pyplot as plt

def galleryplot(func, argnames, iterator, ncols=5, figsize=(4,4), sharex="none", sharey="none"):

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

    W,H = figsize
    iter_for_count = copy.deepcopy(iterator)
    nfigs = len([e for e in iter_for_count])
    nrows = nfigs//ncols if nfigs%ncols==0 else nfigs//ncols+1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=(W*ncols, H*nrows))
    axes = axes if axes.ndim>1 else axes.reshape(1,-1)
    for i,vals in enumerate(iterator):
        kwargs = {argnames:vals} if isinstance(argnames, str) else dict(zip(argnames,vals))
        ax = func(ax=axes[i//ncols, i%ncols], **kwargs)

    return fig, axes
