# coding: utf-8
import re
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib._cm import datad
from mpl_toolkits.mplot3d import Axes3D

from .generic_utils import handleKeyError
from .metric_utils import silhouette_samples
from .metric_utils import silhouette_score

ALL_CMAP_NAMES = [name for name in datad.keys() if not re.search(r".*_r", name, re.IGNORECASE)]
COLOR_TYPES = ["rgba", "rgb", "hex"]

def rgb2hex(rgb, max_val=1):
    return "#"+"".join([format(int(255/max_val*e), '02x') for e in rgb]).upper()

def rgba2rgb(rgba, max_val=1):
    alpha = rgba[-1]
    rgb = rgba[:-1]
    type_ = int if max_val==255 else float
    # compute the color as alpha against white
    return tuple([type_(alpha*e+(1-alpha)*max_val) for e in rgb])

def hex2rgb(hex, max_val=1):
    tuple([int(hex[-6:][i*2:(i+1)*2], 16)/255*max_val for i in range(3)])

def chooseTextColor(rgb, ctype="rgb", max_val=1):
    # Ref: WCAG (https://www.w3.org/TR/WCAG20/)
    R,G,B = [e/max_val for e in rgb]
    # Relative Brightness BackGround.
    Lbg = 0.2126*R + 0.7152*G + 0.0722*B

    Lw = 1 # Relative Brightness of White
    Lb = 0 # Relative Brightness of Black

    Cw = (Lw + 0.05) / (Lbg + 0.05)
    Cb = (Lbg + 0.05) / (Lb + 0.05)
    return (0,0,0) if Cb>Cw else (max_val,max_val,max_val)

def mk_color_dict(keys, cmap="jet", reverse=False, max_val=1, ctype="rgba"):
    """
    @params keys    :
    @params cmap    :
    @params reverse :
    @params max_val :
    @params ctype   : rgb, rgba, hex
    """
    handleKeyError(lst=COLOR_TYPES, ctype=ctype)
    if not isinstance(cmap, colors.Colormap):
        # If `cmap` is string.
        handleKeyError(lst=ALL_CMAP_NAMES, cmap=cmap)
        if reverse:
            cmap += "_r"
        cmap = plt.get_cmap(cmap)

    if isinstance(keys, int):
        N  = keys-1
        color_dict_rgba = {n: cmap(n/N) for n in range(keys)}
    else:
        unique_keys = sorted(list(set(keys)))
        N = len(unique_keys)-1
        color_dict_rgba = {
            key: tuple([
                max_val*e if i<3 else e for i,e in enumerate(cmap(n/N))
            ]) for n,key in enumerate(unique_keys)
        }
    if ctype=="rgba":
        return color_dict_rgba

    color_dict_rgb = {k:rgba2rgb(v, max_val=max_val) for k,v in color_dict_rgba.items()}
    if ctype=="rgb":
        return color_dict_rgb

    if ctype=="hex":
        color_dict_hex = {k:rgb2hex(v[:-1],max_val=max_val) for k,v in color_dict_rgb.items()}
    return color_dict_hex

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
    """Silhouette Plotting.
    @params X       : Input data.               shape=(n_samples, n_features)
    @params labels  : Current label assignment. shape=(n_samples,)
    @params centers : The current centers.      shape=(n_clusters, n_features)
    @params axes    : (axL, axR) if `n_features` <= 3 else ax
    """
    n_features = X.shape[1]
    if n_features <= 3:
        if axes is None:
            fig = plt.figure(figsize=(18, 7))
            projection = "3d" if n_features==3 else None
            axes = (fig.add_subplot(1,2,1), fig.add_subplot(1,2,2,projection=projection))
        ax1 = _left_silhouette_plot(X, labels, ax=axes[0], set_style=set_style)
        ax2 = _right_silhouette_plot(X, labels, centers, ax=axes[1], set_style=set_style)
        axes = (ax1, ax2)
    else:
        if axes is None:
            fig, axes = plt.subplots()
            fig.set_size_inches(18, 7)
        axes = _left_silhouette_plot(X, labels, ax=axes, set_style=set_style)

    if set_style:
        plt.suptitle((f"Silhouette analysis"), fontsize=14, fontweight='bold')
    if axes is None:
        plt.show()
    else:
        return axes

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

        color = cm.hsv(float(i) / n_clusters)
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
        ax.axvline(x=silhouette_avg, color="black", linestyle="--", label="Average")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.legend()
    return ax

def _right_silhouette_plot(X, labels, centers, ax=None, set_style=True):
    n_features = X.shape[1]
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        projection = "3d" if n_features==3 else None
        ax = fig.add_subplot(1,1,1,projection=projection)

    colors = cm.hsv(labels.astype(float) / (np.max(labels)+1.))
    if n_features == 1:
        X_zeros = np.zeros_like(X)
        C_zeros = np.zeros_like(centers)
        ax.scatter(X,       X_zeros, marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
        ax.scatter(centers, C_zeros, marker='o', s=200, alpha=1, c="white", edgecolor='k')
        for i,c in enumerate(centers):
            ax.scatter(c, 0, marker='$%d$' % i, s=50, alpha=1, edgecolor='k')
        plt.tick_params(labelleft=False, left=False)
    elif n_features == 2:
        ax.scatter(X[:,0], X[:,1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
        ax.scatter(centers[:,0], centers[:,1], marker='o', s=200, alpha=1, c="white", edgecolor='k')
        for i,c in enumerate(centers):
            ax.scatter(c[0], c[1], marker='$%d$' % i, s=50, alpha=1, edgecolor='k')
    elif n_features==3:
        ax.scatter(X[:,0], X[:,1], X[:,2], marker='.', s=60, lw=0, alpha=0.7, c=colors, edgecolor='k')
        ax.scatter(centers[:,0], centers[:,1], centers[:,2], marker='o', s=200, alpha=1, c="white", edgecolor='k')
        for i,c in enumerate(centers):
            ax.scatter(c[0], c[1], c[2], marker='$%d$' % i, s=50, alpha=1, edgecolor='k')
    else:
        raise ValueError(f"`X.shape[1]` should be smaller than or equal to 3, but got {n_features}")
    if set_style:
        ax.set_title("The visualization of the clustered data.")
        ax.set_xlabel("Feature space for the 1st feature")
        if n_features>=2: ax.set_ylabel("Feature space for the 2nd feature")
        if n_features==3: ax.set_zlabel("Feature space for the 3rd feature")
    return ax

def plot_CDF(data, ax=None, reverse=False, plot=True, **plotargs):
    """ plot Cumulative Ratio. """
    n_samples = len(data)
    X = sorted(data, reverse=reverse)
    Y = np.arange(1,n_samples+1)/n_samples

    if plot or ax:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(X, Y, **plotargs)
        ax.set_ylabel("Cumulative Ratio")
        return ax

    return (X, Y)

def plot_roc_curve(labels, scores, num=50, ax=None, plot=True, **plotargs):
    flags = labels>0
    tX,tY = plot_CDF(scores[flags], plot=False)
    fX,fY = plot_CDF(scores[~flags], reverse=True, plot=False)
    plot_CDF

    def FRR2score(y):
        idx = np.abs(np.asarray(tY) - y).argmin()
        if y==tY[idx]:
            x=tX[idx]
        else:
            idx = idx-1 if y<tY[idx] else idx
            x = ((y-tY[idx])*tX[idx+1] + (tY[idx+1]-y)*tX[idx])/(tY[idx+1]-tY[idx])
        return x

    def score2FAR(x):
        if x<fX[0]:  return fY[0]
        if x>fX[-1]: return fY[-1]
        idx = np.abs(np.asarray(fX) - x).argmin()
        if x==fX[idx]:
            y=fY[idx]
        else:
            idx = idx-1 if x<fX[idx] else idx
            if fX[idx+1]==fX[idx]:
                y = fY[idx]
            else:
                y = ((x-fX[idx])*fY[idx+1] + (fX[idx+1]-x)*fY[idx])/(fX[idx+1]-fX[idx])
        return y

    FRRs = [i/N for i in range(N+1)]
    FARs = [score2FAR(FRR2score(frr)) for frr in FRRs]
    FARs[0] = 1

    if plot or ax:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(FRRs, FARs, **plotargs)
        ax.set_xlabel("FRR(False Rejection Rate)")
        ax.set_ylabel("FAR(False Acceptance Rate)")
        ax.set_title("The relationship between FRR and FAR")
        return ax

    return (FRRs, FARs)
