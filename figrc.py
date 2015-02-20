# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import sys
sys.path.append(os.getenv('HOME') + '/bin/python/libs')
# just in case notebook was not launched with the option
#%pylab inline

import pylab as plt
import numpy as np
from scipy import sparse
from matplotlib.mlab import griddata
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse

try:
    import faststats
except:
    faststats = None


#===============================================================================
#============== FIGURE SETUP FUNCTIONS =========================================
#===============================================================================
def theme(ax=None, minorticks=False):
    """ update plot to make it nice and uniform """
    from matplotlib.ticker import AutoMinorLocator
    from pylab import rcParams, gca, tick_params
    if minorticks:
        if ax is None:
            ax = gca()
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    tick_params(which='both', width=rcParams['lines.linewidth'])


def steppify(x, y):
    """ Steppify a curve (x,y). Useful for manually filling histograms """
    dx = 0.5 * (x[1:] + x[:-1])
    xx = np.zeros( 2 * len(dx), dtype=float)
    yy = np.zeros( 2 * len(y), dtype=float)
    xx[0::2], xx[1::2] = dx, dx
    yy[0::2], yy[1::2] = y, y
    xx = np.concatenate(([x[0] - (dx[0] - x[0])], xx, [x[-1] + (x[-1] - dx[-1])]))
    return xx, yy


def colorify(data, vmin=None, vmax=None, cmap=plt.cm.Spectral):
    """ Associate a color map to a quantity vector """
    import matplotlib.colors as colors

    _vmin = vmin or min(data)
    _vmax = vmax or max(data)
    cNorm = colors.normalize(vmin=_vmin, vmax=_vmax)

    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    colors = map(scalarMap.to_rgba, data)
    return colors, scalarMap


def devectorize_axes(ax=None, dpi=None, transparent=True):
    """Convert axes contents to a png.

    This is useful when plotting many points, as the size of the saved file
    can become very large otherwise.

    Parameters
    ----------
    ax : Axes instance (optional)
        Axes to de-vectorize.  If None, this uses the current active axes
        (plt.gca())
    dpi: int (optional)
        resolution of the png image.  If not specified, the default from
        'savefig.dpi' in rcParams will be used
    transparent : bool (optional)
        if True (default) then the PNG will be made transparent

    Returns
    -------
    ax : Axes instance
        the in-place modified Axes instance

    Examples
    --------
    The code can be used in the following way::

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        x, y = np.random.random((2, 10000))
        ax.scatter(x, y)
        devectorize_axes(ax)
        plt.savefig('devectorized.pdf')

    The resulting figure will be much smaller than the vectorized version.
    """
    from matplotlib.transforms import Bbox
    from matplotlib import image
    try:
        from io import BytesIO as StringIO
    except ImportError:
        try:
            from cStringIO import StringIO
        except ImportError:
            from StringIO import StringIO

    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    axlim = ax.axis()

    # setup: make all visible spines (axes & ticks) & text invisible
    # we need to set these back later, so we save their current state
    _sp = {}
    _txt_vis = [t.get_visible() for t in ax.texts]
    for k in ax.spines:
        _sp[k] = ax.spines[k].get_visible()
        ax.spines[k].set_visible(False)
    for t in ax.texts:
        t.set_visible(False)

    _xax = ax.xaxis.get_visible()
    _yax = ax.yaxis.get_visible()
    _patch = ax.axesPatch.get_visible()
    ax.axesPatch.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # convert canvas to PNG
    extents = ax.bbox.extents / fig.dpi
    sio = StringIO()
    plt.savefig(sio, format='png', dpi=dpi,
                transparent=transparent,
                bbox_inches=Bbox([extents[:2], extents[2:]]))
    sio.seek(0)
    im = image.imread(sio)

    # clear everything on axis (but not text)
    ax.lines = []
    ax.patches = []
    ax.tables = []
    ax.artists = []
    ax.images = []
    ax.collections = []

    # Show the image
    ax.imshow(im, extent=axlim, aspect='auto', interpolation='nearest')

    # restore all the spines & text
    for k in ax.spines:
        ax.spines[k].set_visible(_sp[k])
    for t, v in zip(ax.texts, _txt_vis):
        t.set_visible(v)
    ax.axesPatch.set_visible(_patch)
    ax.xaxis.set_visible(_xax)
    ax.yaxis.set_visible(_yax)

    if plt.isinteractive():
        plt.draw()

    return ax


def hist_with_err(x, xerr, bins=None, normed=False, step=False, *kwargs):
    from scipy import integrate

    #check inputs
    assert( len(x) == len(xerr) ), 'data size mismatch'
    _x = np.asarray(x).astype(float)
    _xerr = np.asarray(xerr).astype(float)

    #def the evaluation points
    if (bins is None) | (not hasattr(bins, '__iter__')):
        m = (_x - _xerr).min()
        M = (_x + _xerr).max()
        dx = M - m
        m -= 0.2 * dx
        M += 0.2 * dx
        if bins is not None:
            N = int(bins)
        else:
            N = 10
        _xp = np.linspace(m, M, N)
    else:
        _xp = 0.5 * (bins[1:] + bins[:-1])

    def normal(v, mu, sig):
        norm_pdf = 1. / (np.sqrt(2. * np.pi) * sig ) * np.exp( - ( (v - mu ) / (2. * sig) ) ** 2 )
        return norm_pdf / integrate.simps(norm_pdf, v)

    _yp = np.array([normal(_xp, xk, xerrk) for xk, xerrk in zip(_x, _xerr) ]).sum(axis=0)

    if normed:
        _yp /= integrate.simps(_yp, _xp)

    if step:
        return steppify(_xp, _yp)
    else:
        return _xp, _yp


def hist_with_err_bootstrap(x, xerr, bins=None, normed=False, nsample=50, step=False, **kwargs):
    x0, y0 = hist_with_err(x, xerr, bins=bins, normed=normed, step=step, **kwargs)

    yn = np.empty( (nsample, len(y0)), dtype=float)
    yn[0, :] = y0
    for k in range(nsample - 1):
        idx = np.random.randint(0, len(x), len(x))
        yn[k, :] = hist_with_err(x[idx], xerr[idx], bins=bins, normed=normed, step=step, **kwargs)[1]

    return x0, yn


def __get_hesse_bins__(_x, _xerr=0., bins=None, margin=0.2):
    if (bins is None) | (not hasattr(bins, '__iter__')):
        m = (_x - _xerr).min()
        M = (_x + _xerr).max()
        dx = M - m
        m -= margin * dx
        M += margin * dx
        if bins is not None:
            N = int(bins)
        else:
            N = 10
        _xp = np.linspace(m, M, N)
    else:
        _xp = 0.5 * (bins[1:] + bins[:-1])
    return _xp


def scatter_contour(x, y,
                    levels=10,
                    bins=40,
                    threshold=50,
                    log_counts=False,
                    histogram2d_args={},
                    plot_args={},
                    contour_args={},
                    ax=None):
    """Scatter plot with contour over dense regions

    Parameters
    ----------
    x, y : arrays
        x and y data for the contour plot
    levels : integer or array (optional, default=10)
        number of contour levels, or array of contour levels
    threshold : float (default=100)
        number of points per 2D bin at which to begin drawing contours
    log_counts :boolean (optional)
        if True, contour levels are the base-10 logarithm of bin counts.
    histogram2d_args : dict
        keyword arguments passed to numpy.histogram2d
        see doc string of numpy.histogram2d for more information
    plot_args : dict
        keyword arguments passed to pylab.scatter
        see doc string of pylab.scatter for more information
    contourf_args : dict
        keyword arguments passed to pylab.contourf
        see doc string of pylab.contourf for more information
    ax : pylab.Axes instance
        the axes on which to plot.  If not specified, the current
        axes will be used
    """
    if ax is None:
        ax = plt.gca()

    H, xbins, ybins = np.histogram2d(x, y, **histogram2d_args)

    if log_counts:
        H = np.log10(1 + H)
        threshold = np.log10(1 + threshold)

    levels = np.asarray(levels)

    if levels.size == 1:
        levels = np.linspace(threshold, H.max(), levels)

    extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

    i_min = np.argmin(levels)

    # draw a zero-width line: this gives us the outer polygon to
    # reduce the number of points we draw
    # somewhat hackish... we could probably get the same info from
    # the filled contour below.
    outline = ax.contour(H.T, levels[i_min:i_min + 1],
                         linewidths=0, extent=extent)
    try:
        outer_poly = outline.allsegs[0][0]

        ax.contourf(H.T, levels, extent=extent, **contour_args)
        X = np.hstack([x[:, None], y[:, None]])

        try:
            # this works in newer matplotlib versions
            from matplotlib.path import Path
            points_inside = Path(outer_poly).contains_points(X)
        except:
            # this works in older matplotlib versions
            import matplotlib.nxutils as nx
            points_inside = nx.points_inside_poly(X, outer_poly)

        Xplot = X[~points_inside]

        ax.plot(Xplot[:, 0], Xplot[:, 1], zorder=1, **plot_args)
    except IndexError:
        ax.plot(x, y, zorder=1, **plot_args)


def latex_float(f, precision=0.2, delimiter=r'\times'):
    float_str = ("{0:" + str(precision) + "g}").format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return (r"{0}" + delimiter + "10^{{{1}}}").format(base, int(exponent))
    else:
        return float_str


#===============================================================================
#===============================================================================
#===============================================================================

def ezrc(fontSize=22., lineWidth=2., labelSize=None, tickmajorsize=10,
         tickminorsize=5, figsize=(8, 6)):
    """
    slides - Define params to make pretty fig for slides
    """
    from pylab import rc, rcParams
    if labelSize is None:
        labelSize = fontSize + 5
    rc('figure', figsize=figsize)
    rc('lines', linewidth=lineWidth)
    rcParams['grid.linewidth'] = lineWidth
    rcParams['font.sans-serif'] = ['Helvetica']
    rcParams['font.serif'] = ['Helvetica']
    rcParams['font.family'] = ['Times New Roman']
    rc('font', size=fontSize, family='serif', weight='bold')
    rc('axes', linewidth=lineWidth, labelsize=labelSize)
    #rc('xtick', width=2.)
    #rc('ytick', width=2.)
    #rc('legend', fontsize='x-small', borderpad=0.1, markerscale=1.,
    rc('legend', borderpad=0.1, markerscale=1., fancybox=False)
    rc('text', usetex=True)
    rc('image', aspect='auto')
    rc('ps', useafm=True, fonttype=3)
    rcParams['xtick.major.size'] = tickmajorsize
    rcParams['xtick.minor.size'] = tickminorsize
    rcParams['ytick.major.size'] = tickmajorsize
    rcParams['ytick.minor.size'] = tickminorsize
    #rcParams['text.latex.preamble'] = r'\usepackage{pslatex}'
    rcParams['text.latex.preamble'] = ["\\usepackage{amsmath}"]


def hide_axis(where, ax=None):
    ax = ax or plt.gca()
    if type(where) == str:
        _w = [where]
    else:
        _w = where
    [sk.set_color('None') for k, sk in ax.spines.items() if k in _w ]

    if 'top' in _w and 'bottom' in _w:
        ax.xaxis.set_ticks_position('none')
    elif 'top' in _w:
        ax.xaxis.set_ticks_position('bottom')
    elif 'bottom' in _w:
        ax.xaxis.set_ticks_position('top')

    if 'left' in _w and 'right' in _w:
        ax.yaxis.set_ticks_position('none')
    elif 'left' in _w:
        ax.yaxis.set_ticks_position('right')
    elif 'right' in _w:
        ax.yaxis.set_ticks_position('left')

    if ('top' in where) and ('bottom' in where):
        plt.setp(ax.get_xticklabels(), visible=False)
    if ('left' in where) and ('right' in where):
        plt.setp(ax.get_yticklabels(), visible=False)

    plt.draw_if_interactive()


def despine(fig=None, ax=None, top=True, right=True,
            left=False, bottom=False):
    """Remove the top and right spines from plot(s).

    fig : matplotlib figure
        figure to despine all axes of, default uses current figure
    ax : matplotlib axes
        specific axes object to despine
    top, right, left, bottom : boolean
        if True, remove that spine

    """
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            ax_i.spines[side].set_visible(not locals()[side])


def shift_axis(which, delta, where='outward', ax=None):
    ax = ax or plt.gca()
    if type(which) == str:
        _w = [which]
    else:
        _w = which

    scales = (ax.xaxis.get_scale(), ax.yaxis.get_scale())
    lbls = (ax.xaxis.get_label(), ax.yaxis.get_label())

    for wk in _w:
        ax.spines[wk].set_position((where, delta))

    ax.xaxis.set_scale(scales[0])
    ax.yaxis.set_scale(scales[1])
    ax.xaxis.set_label(lbls[0])
    ax.yaxis.set_label(lbls[1])
    plt.draw_if_interactive()


class AutoLocator(MaxNLocator):
    def __init__(self, nbins=9, steps=[1, 2, 5, 10], **kwargs):
        MaxNLocator.__init__(self, nbins=nbins, steps=steps, **kwargs )


def setMargins(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        """
        Tune the subplot layout via the meanings (and suggested defaults) are::

            left  = 0.125  # the left side of the subplots of the figure
            right = 0.9    # the right side of the subplots of the figure
            bottom = 0.1   # the bottom of the subplots of the figure
            top = 0.9      # the top of the subplots of the figure
            wspace = 0.2   # the amount of width reserved for blank space between subplots
            hspace = 0.2   # the amount of height reserved for white space between subplots

        The actual defaults are controlled by the rc file

        """
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        plt.draw_if_interactive()


def setNmajors(xval=None, yval=None, ax=None, mode='auto', **kwargs):
        """
        setNmajors - set major tick number
        see figure.MaxNLocator for kwargs
        """
        if ax is None:
                ax = plt.gca()
        if (mode == 'fixed'):
                if xval is not None:
                        ax.xaxis.set_major_locator(MaxNLocator(xval, **kwargs))
                if yval is not None:
                        ax.yaxis.set_major_locator(MaxNLocator(yval, **kwargs))
        elif (mode == 'auto'):
                if xval is not None:
                        ax.xaxis.set_major_locator(AutoLocator(xval, **kwargs))
                if yval is not None:
                        ax.yaxis.set_major_locator(AutoLocator(yval, **kwargs))

        plt.draw_if_interactive()


def crazy_histogram2d(x, y, bins=10, weights=None, reduce_w=None, NULL=None, reinterp=None):
    """
    Compute the sparse bi-dimensional histogram of two data samples where *x*,
    and *y* are 1-D sequences of the same length. If *weights* is None
    (default), this is a histogram of the number of occurences of the
    observations at (x[i], y[i]).

    If *weights* is specified, it specifies values at the coordinate (x[i],
    y[i]). These values are accumulated for each bin and then reduced according
    to *reduce_w* function, which defaults to numpy's sum function (np.sum).
    (If *weights* is specified, it must also be a 1-D sequence of the same
    length as *x* and *y*.)

    INPUTS:
        x       ndarray[ndim=1]         first data sample coordinates
        y       ndarray[ndim=1]         second data sample coordinates

    KEYWORDS:
        bins                            the bin specification
                   int                     the number of bins for the two dimensions (nx=ny=bins)
                or [int, int]              the number of bins in each dimension (nx, ny = bins)
        weights     ndarray[ndim=1]     values *w_i* weighing each sample *(x_i, y_i)*
                                        accumulated and reduced (using reduced_w) per bin
        reduce_w    callable            function that will reduce the *weights* values accumulated per bin
                                        defaults to numpy's sum function (np.sum)
        NULL        value type          filling missing data value
        reinterp    str                 values are [None, 'nn', linear']
                                        if set, reinterpolation is made using mlab.griddata to fill missing data
                                        within the convex polygone that encloses the data

    OUTPUTS:
        B           ndarray[ndim=2]     bi-dimensional histogram
        extent      tuple(4)            (xmin, xmax, ymin, ymax) entension of the histogram
        steps       tuple(2)            (dx, dy) bin size in x and y direction

    """
    # define the bins (do anything you want here but needs edges and sizes of the 2d bins)
    try:
        nx, ny = bins
    except TypeError:
        nx = ny = bins

    #values you want to be reported
    if weights is None:
        weights = np.ones(x.size)

    if reduce_w is None:
        reduce_w = np.sum
    else:
        if not hasattr(reduce_w, '__call__'):
            raise TypeError('reduce function is not callable')

    # culling nans
    finite_inds = (np.isfinite(x) & np.isfinite(y) & np.isfinite(weights))
    _x = np.asarray(x)[finite_inds]
    _y = np.asarray(y)[finite_inds]
    _w = np.asarray(weights)[finite_inds]

    if not (len(_x) == len(_y)) & (len(_y) == len(_w)):
        raise ValueError('Shape mismatch between x, y, and weights: {}, {}, {}'.format(_x.shape, _y.shape, _w.shape))

    xmin, xmax = _x.min(), _x.max()
    ymin, ymax = _y.min(), _y.max()
    dx = (xmax - xmin) / (nx - 1.0)
    dy = (ymax - ymin) / (ny - 1.0)

    # Basically, this is just doing what np.digitize does with one less copy
    xyi = np.vstack((_x, _y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    #xyi contains the bins of each point as a 2d array [(xi,yi)]

    d = {}
    for e, k in enumerate(xyi.T):
        key = (k[0], k[1])

        if key in d:
            d[key].append(_w[e])
        else:
            d[key] = [_w[e]]

    _xyi = np.array(d.keys()).T
    _w   = np.array([ reduce_w(v) for v in d.values() ])

    # exploit a sparse coo_matrix to build the 2D histogram...
    _grid = sparse.coo_matrix((_w, _xyi), shape=(nx, ny))

    if reinterp is None:
        #convert sparse to array with filled value
        ## grid.toarray() does not account for filled value
        ## sparse.coo.coo_todense() does actually add the values to the existing ones, i.e. not what we want -> brute force
        if NULL is None:
            B = _grid.toarray()
        else:  # Brute force only went needed
            B = np.zeros(_grid.shape, dtype=_grid.dtype)
            B.fill(NULL)
            for (x, y, v) in zip(_grid.col, _grid.row, _grid.data):
                B[y, x] = v
    else:  # reinterp
        xi = np.arange(nx, dtype=float)
        yi = np.arange(ny, dtype=float)
        B = griddata(_grid.col.astype(float), _grid.row.astype(float), _grid.data, xi, yi, interp=reinterp)

    return B, (xmin, xmax, ymin, ymax), (dx, dy)


def histplot(data, bins=10, range=None, normed=False, weights=None, density=None, ax=None, **kwargs):
    """ plot an histogram of data `a la R`: only bottom and left axis, with
    dots at the bottom to represent the sample

    Example
    -------
        import numpy as np
        x = np.random.normal(0, 1, 1e3)
        histplot(x, bins=50, density=True, ls='steps-mid')
    """
    h, b = np.histogram(data, bins, range, normed, weights, density)
    if ax is None:
        ax = plt.gca()
    x = 0.5 * (b[:-1] + b[1:])
    l = ax.plot(x, h, **kwargs)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    _w = ['top', 'right']
    [ ax.spines[side].set_visible(False) for side in _w ]

    for wk in ['bottom', 'left']:
        ax.spines[wk].set_position(('outward', 10))

    ylim = ax.get_ylim()
    ax.plot(data, -0.02 * max(ylim) * np.ones(len(data)), '|', color=l[0].get_color())
    ax.set_ylim(-0.02 * max(ylim), max(ylim))


def scatter_plot(x, y, ellipse=False, levels=[0.99, 0.95, 0.68], color='w', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    if faststats is not None:
        im, e = faststats.fastkde.fastkde(x, y, (50, 50), adjust=2.)
        V = im.max() * np.asarray(levels)

        plt.contour(im.T, levels=V, origin='lower', extent=e, linewidths=[1, 2, 3], colors=color)

    ax.plot(x, y, 'b,', alpha=0.3, zorder=-1, rasterized=True)

    if ellipse is True:
        data = np.vstack([x, y])
        mu = np.mean(data, axis=1)
        cov = np.cov(data)
        error_ellipse(mu, cov, ax=plt.gca(), edgecolor="g", ls="dashed", lw=4, zorder=2)


def error_ellipse(mu, cov, ax=None, factor=1.0, **kwargs):
    """
    Plot the error ellipse at a point given its covariance matrix.

    """
    # some sane defaults
    facecolor = kwargs.pop('facecolor', 'none')
    edgecolor = kwargs.pop('edgecolor', 'k')

    x, y = mu
    U, S, V = np.linalg.svd(cov)
    theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    ellipsePlot = Ellipse(xy=[x, y],
                          width=2 * np.sqrt(S[0]) * factor,
                          height=2 * np.sqrt(S[1]) * factor,
                          angle=theta,
                          facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_patch(ellipsePlot)

    return ellipsePlot


def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]


def quantiles(x, qlist=[2.5, 25, 50, 75, 97.5]):
    """computes quantiles from an array

    Quantiles :=  points taken at regular intervals from the cumulative
    distribution function (CDF) of a random variable. Dividing ordered data
    into q essentially equal-sized data subsets is the motivation for
    q-quantiles; the quantiles are the data values marking the boundaries
    between consecutive subsets.

    The quantile with a fraction 50 is called the median
    (50% of the distribution)

    Inputs:
        x     - variable to evaluate from
        qlist - quantiles fraction to estimate (in %)

    Outputs:
        Returns a dictionary of requested quantiles from array
    """
    # Make a copy of trace
    x = x.copy()

    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort, then transpose back
        sx = np.transpose(np.sort(np.transpose(x)))
    else:
        # Sort univariate node
        sx = np.sort(x)

    try:
        # Generate specified quantiles
        quants = [sx[int(len(sx) * q / 100.0)] for q in qlist]

        return dict(zip(qlist, quants))

    except IndexError:
        print("Too few elements for quantile calculation")


def get_optbins(data, method='freedman', ret='N'):
    """ Determine the optimal binning of the data based on common estimators
    and returns either the number of bins of the width to use.

    input:
        data    1d dataset to estimate from
    keywords:
        method  the method to use: str in {sturge, scott, freedman}
        ret set to N will return the number of bins / edges
            set to W will return the width
    refs:
        Sturges, H. A. (1926)."The choice of a class interval". J. American Statistical Association, 65-66
        Scott, David W. (1979), "On optimal and data-based histograms". Biometrika, 66, 605-610
        Freedman, D.; Diaconis, P. (1981). "On the histogram as a density estimator: L2 theory".
                Zeitschrift fur Wahrscheinlichkeitstheorie und verwandte Gebiete, 57, 453-476
        Scargle, J.D. et al (2012) "Studies in Astronomical Time Series Analysis. VI. Bayesian
        Block Representations."
    """
    x = np.asarray(data)
    n = x.size
    r = x.max() - x.min()

    def sturge():
        if (n <= 30):
            print("Warning: Sturge estimator can perform poorly for small samples")
        k = int(np.log(n) + 1)
        h = r / k
        return h, k

    def scott():
        h = 3.5 * np.std(x) * float(n) ** (-1. / 3.)
        k = int(r / h)
        return h, k

    def freedman():
        q = quantiles(x, [25, 75])
        h = 2 * (q[75] - q[25]) * float(n) ** (-1. / 3.)
        k = int(r / h)
        return h, k

    def bayesian():
        r = bayesian_blocks(x)
        return np.diff(r),r

    m = {'sturge': sturge, 'scott': scott, 'freedman': freedman,
         'bayesian': bayesian}

    if method.lower() in m:
        s = m[method.lower()]()
        if ret.lower() == 'n':
            return s[1]
        elif ret.lower() == 'w':
            return s[0]
    else:
        return None


def plotMAP(x, ax=None, error=0.01, frac=[0.65,0.95, 0.975], usehpd=True,
            hist={'histtype':'step'}, vlines={}, fill={},
            optbins={'method':'freedman'}, *args, **kwargs):
    """ Plot the MAP of a given sample and add statistical info
    If not specified, binning is assumed from the error value or using
    mystats.optbins if available.
    if mystats module is not available, hpd keyword has no effect

    inputs:
        x   dataset
    keywords
        ax  axe object to use during plotting
        error   error to consider on the estimations
        frac    fractions of sample to highlight (def 65%, 95%, 97.5%)
        hpd if set, uses mystats.hpd to estimate the confidence intervals

        hist    keywords forwarded to hist command
        optbins keywords forwarded to mystats.optbins command
        vlines  keywords forwarded to vlines command
        fill    keywords forwarded to fill command
        """
    _x = np.ravel(x)
    if ax is None:
        ax = plt.gca()
    if not ('bins' in hist):
        bins = get_optbins(x, method=optbins['method'], ret='N')
        n, b, p = ax.hist(_x, bins=bins, *args, **hist)
    else:
        n, b, p = ax.hist(_x, *args, **hist)
    c = 0.5 * (b[:-1] + b[1:])
    #dc = 0.5 * (b[:-1] - b[1:])
    ind = n.argmax()
    _ylim = ax.get_ylim()
    if usehpd is True:
        _hpd = hpd(_x, 1 - 0.01)
        ax.vlines(_hpd, _ylim[0], _ylim[1], **vlines)
        for k in frac:
            nx = hpd(_x, 1. - k)
            ax.fill_between(nx, _ylim[0], _ylim[1], alpha=0.4 / float(len(frac)), zorder=-1, **fill)
    else:
        ax.vlines(c[ind], _ylim[0], _ylim[1], **vlines)
        cx = c[ n.argsort() ][::-1]
        cn = n[ n.argsort() ][::-1].cumsum()
        for k in frac:
            sx = cx[np.where(cn <= cn[-1] * float(k))]
            sx = [sx.min(), sx.max()]
            ax.fill_between(sx, _ylim[0], _ylim[1], alpha=0.4 / float(len(frac)), zorder=-1, **fill)
    theme(ax=ax)
    ax.set_xlabel(r'Values')
    ax.set_ylabel(r'Counts')


def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width"""

    # Initialize interval
    min_int = [None, None]

    try:

        # Number of elements in trace
        n = len(x)

        # Start at far left
        start, end = 0, int(n * (1 - alpha))

        # Initialize minimum width to large value
        min_width = np.inf

        while end < n:

            # Endpoints of interval
            hi, lo = x[end], x[start]

            # Width of interval
            width = hi - lo

            # Check to see if width is narrower than minimum
            if width < min_width:
                min_width = width
                min_int = [lo, hi]

            # Increment endpoints
            start += 1
            end += 1

        return min_int

    except IndexError:
        print('Too few elements for interval calculation')
        return [None, None]


def getPercentileLevels(h, frac=[0.5, 0.65, 0.95, 0.975]):
    """
    Return image levels that corresponds to given percentiles values
    Uses the cumulative distribution of the sorted image density values
    Hence this works also for any nd-arrays
    inputs:
        h   array
    outputs:
        res array containing level values
    keywords:
        frac    sample fractions (percentiles)
            could be scalar or iterable
            default: 50%, 65%, 95%, and 97.5%

    """
    if getattr(frac, '__iter__', False):
        return np.asarray( [getPercentileLevels(h, fk) for fk in frac])

    if not ((frac >= 0.) & (frac < 1.)):
        raise ValueError("Expecting a sample fraction in 'frac' and got %f" % frac)

    # flatten the array to a 1d list
    val = h.ravel()
    # inplace sort
    val.sort()
    #reverse order
    rval = val[::-1]
    #cumulative values
    cval = rval.cumsum()
    cval = (cval - cval[0]) / (cval[-1] - cval[0])
    #retrieve the largest indice up to the fraction of the sample we want
    ind = np.where(cval <= cval[-1] * float(frac))[0].max()
    res = rval[ind]
    del val, cval, ind, rval
    return res


def plotDensity(x,y, bins=100, ax=None, Nlevels=None, levels=None,
                frac=None,
                contour={'colors':'0.0', 'linewidths':0.5},
                contourf={'cmap': plt.cm.Greys_r},
                scatter={'c':'0.0', 's':0.5, 'edgecolor':'None'},
                *args, **kwargs ):
    """
    Plot a the density of x,y given certain contour paramters and includes
    individual points (not represented by contours)

    inputs:
        x,y data to plot

    keywords:
        bins    bin definition for the density histogram
        ax  use a specific axis
        Nlevels the number of levels to use with contour
        levels  levels
        frac    percentiles to contour if specified

        Extra keywords:
        *args, **kwargs forwarded to histogram2d
        **contour       forwarded to contour function
        **contourf      forwarded to contourf function
        **plot          forwarded to contourf function

    """
    if ax is None:
        ax = plt.gca()

    if 'bins' not in kwargs:
        kwargs['bins'] = bins

    h, xe, ye = np.histogram2d(x, y, *args, **kwargs)

    if (Nlevels is None) & (levels is None) & (frac is None):
        levels = np.sort(getPercentileLevels(h))
    elif (Nlevels is not None) & (levels is None) & (frac is None):
        levels = np.linspace(2., h.max(), Nlevels)[1:].tolist() + [h.max()]
    elif (frac is not None):
        levels = getPercentileLevels(h, frac=frac)

    if not getattr(levels, '__iter__', False):
        raise AttributeError("Expecting levels variable to be iterable")

    if levels[-1] != h.max():
        levels = list(levels) + [h.max()]

    if isinstance(contourf, dict):
        cont = ax.contourf(h.T, extent=[xe[0],xe[-1], ye[0],ye[-1]],
                           levels=levels, **contourf)
    else:
        cont = None

    if isinstance(contour, dict):
        ax.contour(h.T, extent=[xe[0],xe[-1], ye[0],ye[-1]], levels=levels,
                   **contour)

    ind = np.asarray([False] * len(x))

    if cont is not None:
        nx = np.ceil(np.interp(x, 0.5 * (xe[:-1] + xe[1:]), range(len(xe) - 1)))
        ny = np.ceil(np.interp(y, 0.5 * (ye[:-1] + ye[1:]), range(len(ye) - 1)))
        nh = [ h[nx[k],ny[k]] for k in range(len(x)) ]
        ind = np.where(nh < np.min(levels))
        ax.scatter(x[ind], y[ind], **scatter)
    else:
        ax.plot(x, y, **scatter)


def make_indices(dimensions):
    """ Generates complete set of indices for given dimensions """

    level = len(dimensions)

    if level == 1:
        return range(dimensions[0])

    indices = [[]]

    while level:

        _indices = []

        for j in range(dimensions[level - 1]):

            _indices += [[j] + i for i in indices]

        indices = _indices

        level -= 1

    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices


def hpd(x, alpha, copy=True):
    """Calculate HPD (minimum width BCI) of array for given alpha"""

    if hasattr(alpha, '__iter__'):
        return np.array([ hpd(x, ak, copy=copy) for ak in alpha ])

    # Make a copy of trace
    if copy is True:
        x = x.copy()

    # Transpose first, then sort
    tx = np.transpose(x, list(range(x.ndim)[1:]) + [0])
    dims = np.shape(tx)

    # Container list for intervals
    intervals = np.resize(0.0, dims[:-1] + (2,))

    for index in make_indices(dims[:-1]):

        try:
            index = tuple(index)
        except TypeError:
            pass

        # Sort trace
        sx = np.sort(tx[index])

        # Append to list
        intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)


def plotCorr(l, pars, plotfunc=None, lbls=None, limits=None, triangle='lower',
             devectorize=False, *args, **kwargs):
        """ Plot correlation matrix between variables
        inputs
        -------
        l: dict
            dictionary of variables (could be a Table)

        pars: sequence of str
            parameters to use

        plotfunc: callable
            function to be called when doing the scatter plots

        lbls: sequence of str
            sequence of string to use instead of dictionary keys

        limits: dict
            impose limits for some paramters. Each limit should be pairs of values.
            No need to define each parameter limits

        triangle: str in ['upper', 'lower']
            Which side of the triangle to use.

        devectorize: bool
            if set, rasterize the figure to reduce its size

        *args, **kwargs are forwarded to the plot function

        Example
        -------
            import numpy as np
            figrc.ezrc(16, 1, 16, 5)

            d = {}

            for k in range(4):
                d[k] = np.random.normal(0, k+1, 1e4)

            plt.figure(figsize=(8 * 1.5, 7 * 1.5))
            plotCorr(d, d.keys(), plotfunc=figrc.scatter_plot)
            #plotCorr(d, d.keys(), alpha=0.2)
        """

        if lbls is None:
                lbls = pars

        if limits is None:
            limits = {}

        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}
        if not len(pars) - 1 in fontmap:
                fontmap[len(pars) - 1] = 3

        k = 1
        axes = np.empty((len(pars) + 1, len(pars)), dtype=object)
        for j in range(len(pars)):
                for i in range(len(pars)):
                        if j > i:
                                sharex = axes[j - 1, i]
                        else:
                                sharex = None

                        if i == j:
                            # Plot the histograms.
                            ax = plt.subplot(len(pars), len(pars), k)
                            axes[j, i] = ax
                            data = l[pars[i]]
                            n, b, p = ax.hist(data, bins=50, histtype="step", color=kwargs.get("color", "k"))
                            if triangle == 'upper':
                                ax.set_xlabel(lbls[i])
                                ax.set_ylabel(lbls[i])
                                ax.xaxis.set_ticks_position('bottom')
                                ax.yaxis.set_ticks_position('none')
                            else:
                                ax.yaxis.set_ticks_position('none')
                                ax.xaxis.set_ticks_position('bottom')
                                hide_axis(['right', 'top', 'left'], ax=ax)
                                plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)

                            xlim = limits.get(pars[i], (data.min(), data.max()))
                            ax.set_xlim(xlim)
                            #ax.set_ylim(0, n.max() * 1.1)

                        if triangle == 'upper':
                            data_x = l[pars[i]]
                            data_y = l[pars[j]]
                            if i > j:

                                if i > j + 1:
                                        sharey = axes[j, i - 1]
                                else:
                                        sharey = None

                                ax = plt.subplot(len(pars), len(pars), k, sharey=sharey, sharex=sharex)
                                axes[j, i] = ax
                                if plotfunc is None:
                                        plt.plot(data_x, data_y, ',', **kwargs)
                                else:
                                        plotfunc(data_x, data_y, ax=ax, *args, **kwargs)
                                xlim = limits.get(pars[i], None)
                                ylim = limits.get(pars[j], None)
                                if xlim is not None:
                                    ax.set_xlim(xlim)
                                if ylim is not None:
                                    ax.set_ylim(ylim)

                                plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)
                                if devectorize is True:
                                    devectorize_axes(ax=ax)

                        if triangle == 'lower':
                            data_x = l[pars[i]]
                            data_y = l[pars[j]]
                            if i < j:

                                if i < j:
                                        sharey = axes[j, i - 1]
                                else:
                                        sharey = None

                                ax = plt.subplot(len(pars), len(pars), k, sharey=sharey, sharex=sharex)
                                axes[j, i] = ax
                                if plotfunc is None:
                                        plt.plot(data_x, data_y, ',', **kwargs)
                                else:
                                        plotfunc(data_x, data_y, ax=ax, *args, **kwargs)

                                plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)
                                xlim = limits.get(pars[i], None)
                                ylim = limits.get(pars[j], None)
                                if xlim is not None:
                                    ax.set_ylim(xlim)
                                if ylim is not None:
                                    ax.set_ylim(ylim)
                                if devectorize is True:
                                    devectorize_axes(ax=ax)

                            if i == 0:
                                ax.set_ylabel(lbls[j])
                                plt.setp(ax.get_yticklabels(), visible=True)

                            if j == len(pars) - 1:
                                ax.set_xlabel(lbls[i])
                                plt.setp(ax.get_xticklabels(), visible=True)

                        N = int(0.5 * fontmap[len(pars) - 1])
                        if N <= 4:
                            N = 5
                        setNmajors(N, N, ax=ax, prune='both')

                        k += 1
        setMargins(hspace=0.0, wspace=0.0)


def hinton(W, bg='grey', facecolors=('w', 'k')):
    """Draw a hinton diagram of the matrix W on the current pylab axis

    Hinton diagrams are a way of visualizing numerical values in a matrix/vector,
    popular in the neural networks and machine learning literature. The area
    occupied by a square is proportional to a value's magnitude, and the colour
    indicates its sign (positive/negative).

    Example usage:

        R = np.random.normal(0, 1, (2,1000))
        h, ex, ey = np.histogram2d(R[0], R[1], bins=15)
        hh = h - h.T
        hinton.hinton(hh)
    """
    M, N = W.shape
    square_x = np.array([-.5, .5, .5, -.5])
    square_y = np.array([-.5, -.5, .5, .5])

    ioff = False
    if plt.isinteractive():
        plt.ioff()
        ioff = True

    plt.fill([-.5, N - .5, N - .5, - .5], [-.5, -.5, M - .5, M - .5], bg)
    Wmax = np.abs(W).max()
    for m, Wrow in enumerate(W):
        for n, w in enumerate(Wrow):
            c = plt.signbit(w) and facecolors[1] or facecolors[0]
            plt.fill(square_x * w / Wmax + n, square_y * w / Wmax + m, c, edgecolor=c)

    plt.ylim(-0.5, M - 0.5)
    plt.xlim(-0.5, M - 0.5)

    if ioff is True:
        plt.ion()

    plt.draw_if_interactive()


def parallel_coordinates(d, labels=None, orientation='horizontal',
                         positions=None, ax=None, **kwargs):
    """ Plot parallel coordinates of a data set

    Each dimension is normalized and then plot either vertically or horizontally

    Parameters
    ----------
    d: ndarray, recarray or dict
        data to plot (one column or key per coordinate)

    labels: sequence
        sequence of string to use to define the label of each coordinate
        default p{:d}

    orientation: str
        'horizontal' of 'vertical' to set the plot orientation accordingly

    positions: sequence(float)
        position of each plane on the main axis. Default is equivalent to
        equidistant positioning.

    ax: plt.Axes instance
        axes to use for the figure, default plt.subplot(111)

    **kwargs: dict
        forwarded to :func:`plt.plot`
    """

    if labels is None:
        if hasattr(d, 'keys'):
            names = list(d.keys())
            data = np.array(d.values()).T
        elif hasattr(d, 'dtype'):
            if d.dtype.names is not None:
                names = d.dtype.names
            else:
                names = [ 'p{0:d}'.format(k) for k in range(len(d[0])) ]
        else:
                names = [ 'p{0:d}'.format(k) for k in range(len(d)) ]
        data = np.array(d).astype(float)
    else:
        names = labels
        data = np.array(d).astype(float)

    if len(labels) != len(data[0]):
        names = [ 'p{0:d}'.format(k) for k in range(len(data[0])) ]

    if positions is None:
        positions = np.arange(len(names))
    else:
        positions = np.array(positions)

    positions -= positions.min()
    dyn = np.ptp(positions)

    data = (data - data.mean(axis=0)) / data.ptp(axis=0)[None, :]
    order = np.argsort(positions)
    data = data[:, order]
    positions = positions[order]
    names = np.array(names)[order].tolist()

    if ax is None:
        ax = plt.subplot(111)

    if orientation.lower() == 'horizontal':
        ax.vlines(positions, -1, 1, color='0.8')
        ax.plot(positions, data.T, **kwargs)
        hide_axis(['left', 'right', 'top', 'bottom'], ax=ax)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xticks(positions)
        ax.set_xticklabels(names)
        ax.set_xlim(positions.min() - 0.1 * dyn, positions.max() + 0.1 * dyn)
    else:
        ax.hlines(positions, -1, 1, color='0.8')
        ax.plot(data.T, positions, **kwargs)
        hide_axis(['left', 'right', 'top', 'bottom'], ax=ax)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_yticks(positions)
        ax.set_yticklabels(names)
        ax.set_ylim(positions.min() - 0.1 * dyn, positions.max() + 0.1 * dyn)
