r""" Fitting a straight line with non-symmetric (but Gaussian) uncertainties

.. Example::

    >>> python btmc.py <file> -N 200 -n 10 -c <default.cfg>

    >>> python btmc.py <file> -N 200 -n 1\
           --xnorm 6 --ynorm 0.5 --xfloor 10 --yfloor 10
"""
# make this works for both python 2 and python 3
from __future__ import print_function
# numerical package
import numpy as np
# plotting package
import pylab as plt
from scipy.sparse import coo_matrix
from scipy.signal import convolve2d

# my own set of plotting tools (more like for polishing)
try:
    import figrc
    figrc.ezrc(12, 1, 16, 5, 2)
except ImportError:
    figrc = None


import sys
PY3 = sys.version_info[0] > 2

if PY3:
    basestring = (str, bytes)
else:
    range = xrange
    basestring = (str, unicode)

import matplotlib.patheffects as PathEffects


def MAP(x, y, xmin, xmax, ymin, ymax, nsamp=1000, xnorm=6, ynorm=0.5,
        errorfloor=1e-4, autonorm=True):
    """
    Resample the data to approximate its noisy distribution.

    Parameters
    ----------
    x: ndarray
        x-coordinates

    y: ndarray
        y-coordinates

    xmin: ndarray
        lower uncertainty on x

    xmax: ndarray
        upper uncertainty on x

    ymin: ndarray
        lower uncertainty on y

    ymax: ndarray
        upper uncertainty on y

    nsamp: int, optional
        number of samples to draw from the uncertainties (per input point)

    xnorm: float
        x-normalization of the fit (data normalization)

    ynorm: float
        y-normalization of the fit (data normalization)

    errorfloor: float, optional
        add a small sigma to the random draws to avaid issues with null uncertainties

    Returns
    -------
    alpha: float
        MAP slope of the model

    beta: float
        MAP intercept of the model

    d: float
        dispersion around the model
    """

    xdata, ydata = sample_data(x, y, xmin, xmax, ymin, ymax, nsamp=nsamp,
                               errorfloor=errorfloor)

    xdata = np.array(xdata).ravel()
    ydata = np.array(ydata).ravel()

    ind = np.isfinite(xdata) & np.isfinite(ydata) & (xdata > 0) & (ydata > 0)
    xdata = np.log10(xdata[ind])
    ydata = np.log10(ydata[ind])

    alpha, beta, delta = OLS_MAP(xdata - xnorm, ydata - ynorm)

    return alpha, beta, delta


def sample_data(x, y, xmin, xmax, ymin, ymax, nsamp=1000, errorfloor=1e-4):
    """
    Resample the data to approximate its noisy distribution.

    Parameters
    ----------
    x: ndarray
        x-coordinates

    y: ndarray
        y-coordinates

    xmin: ndarray
        lower uncertainty on x

    xmax: ndarray
        upper uncertainty on x

    ymin: ndarray
        lower uncertainty on y

    ymax: ndarray
        upper uncertainty on y

    nsamp: int, optional
        number of samples to draw from the uncertainties (per input point)

    errorfloor: float, optional
        add a small sigma to the random draws to avaid issues with null uncertainties

    Returns
    -------
    xdata: ndarray, shape = Nsamp , Npts
        x sequence resampled from errors

    ydata: ndarray, shape = Nsamp , Npts
        y sequence resampled from errors
    """
    xdata = []
    ydata = []

    for k in range(len(x)):
        xdata.append(x[k] + np.abs(np.random.normal(0, xmax[k] + errorfloor, nsamp)))
        ydata.append(y[k] + np.abs(np.random.normal(0, ymax[k] + errorfloor, nsamp)))

        xdata.append(x[k] - np.abs(np.random.normal(0, xmin[k] + errorfloor, nsamp)))
        ydata.append(y[k] - np.abs(np.random.normal(0, ymin[k] + errorfloor, nsamp)))

        xdata.append(x[k] + np.abs(np.random.normal(0, xmax[k] + errorfloor, nsamp)))
        ydata.append(y[k] - np.abs(np.random.normal(0, ymin[k] + errorfloor, nsamp)))

        xdata.append(x[k] - np.abs(np.random.normal(0, xmin[k] + errorfloor, nsamp)))
        ydata.append(y[k] + np.abs(np.random.normal(0, ymax[k] + errorfloor, nsamp)))

    return xdata, ydata


def OLS_MAP(X, Y):
    """ Perpendicular/Orthogonal least square maximum a posteriori

    .. math::

        y( x | a, b) = a x + b

    Parameters
    ----------
    X: ndarray
        x sequence

    Y: ndarray
        y sequence

    Returns
    -------
    a: float
        slope

    b: float
        intercept

    d: float
        dispersion around the model

    .. note::

        http://mathworld.wolfram.com/LeastSquaresFittingPerpendicularOffsets.html
    """
    ind = np.isfinite(X) & np.isfinite(Y)
    x = X[ind]
    y = Y[ind]
    n = len(x)
    c = np.cov(x, y)
    sxx = n * c[0, 0]
    syy = n * c[1, 1]
    sxy = (n - 1) * np.cov(x, y)[1, 0]

    a = 0.5 * (syy - sxx + np.sqrt( (syy - sxx) ** 2 + 4 * sxy ** 2) ) / sxy
    b = y.mean() - a * x.mean()

    # trying to get residual perpendicularly to the line
    v = 1. / np.sqrt( 1. + a ** 2) * np.array([-a, 1 ])
    vects = np.vstack([x - x.mean(), y - y.mean()]).T
    delta = np.sqrt((np.array([np.dot(v, k) for k in vects]) ** 2).sum())

    return a, b, (delta / len(x - 1)) ** 0.5


def __calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width
    """

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


def __make_indices(dimensions):
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


def hpd(x, alpha):
    """Calculate HPD (minimum width BCI) of array for given alpha

    Parameters
    ----------
    x: ndarray
        one random variable

    alpha: sequence or float
        CI level

    Returns
    -------
    vals: sequence
        sequence of intervals (one interval per alpha)
    """

    if hasattr(alpha, '__iter__'):
        return np.array([ hpd(x, ak) for ak in alpha ])

    # Transpose first, then sort
    tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
    dims = np.shape(tx)

    # Container list for intervals
    intervals = np.resize(0.0, dims[:-1] + (2,))

    for index in __make_indices(dims[:-1]):

        try:
            index = tuple(index)
        except TypeError:
            pass

        # Sort trace
        sx = np.sort(tx[index])

        # Append to list
        intervals[index] = __calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)


def fastkde(x, y, gridsize=(200, 200), extents=None, nocorrelation=False,
            weights=None, adjust=1.):
    """
    A fft-based Gaussian kernel density estimate (KDE)
    for computing the KDE on a regular grid

    Note that this is a different use case than scipy's original
    scipy.stats.kde.gaussian_kde

    IMPLEMENTATION
    --------------

    Performs a gaussian kernel density estimate over a regular grid using a
    convolution of the gaussian kernel with a 2D histogram of the data.

    It computes the sparse bi-dimensional histogram of two data samples where
    *x*, and *y* are 1-D sequences of the same length. If *weights* is None
    (default), this is a histogram of the number of occurences of the
    observations at (x[i], y[i]).
    histogram of the data is a faster implementation than numpy.histogram as it
    avoids intermediate copies and excessive memory usage!


    This function is typically *several orders of magnitude faster* than
    scipy.stats.kde.gaussian_kde.  For large (>1e7) numbers of points, it
    produces an essentially identical result.

    Boundary conditions on the data is corrected by using a symmetric /
    reflection condition. Hence the limits of the dataset does not affect the
    pdf estimate.

    Parameters
    ----------

        x, y:  ndarray[ndim=1]
            The x-coords, y-coords of the input data points respectively

        gridsize: tuple
            A (nx,ny) tuple of the size of the output grid (default: 200x200)

        extents: (xmin, xmax, ymin, ymax) tuple
            tuple of the extents of output grid (default: extent of input data)

        nocorrelation: bool
            If True, the correlation between the x and y coords will be ignored
            when preforming the KDE. (default: False)

        weights: ndarray[ndim=1]
            An array of the same shape as x & y that weights each sample (x_i,
            y_i) by each value in weights (w_i).  Defaults to an array of ones
            the same size as x & y. (default: None)

        adjust : float
            An adjustment factor for the bw. Bandwidth becomes bw * adjust.

    Returns
    -------
        g: ndarray[ndim=2]
            A gridded 2D kernel density estimate of the input points.

        e: (xmin, xmax, ymin, ymax) tuple
            Extents of g

    """
    # Variable check
    x, y = np.asarray(x), np.asarray(y)
    x, y = np.squeeze(x), np.squeeze(y)

    if x.size != y.size:
        raise ValueError('Input x & y arrays must be the same size!')

    n = x.size

    if weights is None:
        # Default: Weight all points equally
        weights = np.ones(n)
    else:
        weights = np.squeeze(np.asarray(weights))
        if weights.size != x.size:
            raise ValueError('Input weights must be an array of the same size as input x & y arrays!')

    # Optimize gridsize ------------------------------------------------------
    # Make grid and discretize the data and round it to the next power of 2
    # to optimize with the fft usage
    if gridsize is None:
        gridsize = np.asarray([np.max((len(x), 512.)), np.max((len(y), 512.))])
    gridsize = 2 ** np.ceil(np.log2(gridsize))  # round to next power of 2

    nx, ny = gridsize

    # Make the sparse 2d-histogram -------------------------------------------
    # Default extents are the extent of the data
    if extents is None:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
    else:
        xmin, xmax, ymin, ymax = map(float, extents)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # Basically, this is just doing what np.digitize does with one less copy
    # xyi contains the bins of each point as a 2d array [(xi,yi)]
    xyi = np.vstack((x,y)).T
    xyi -= [xmin, ymin]
    xyi /= [dx, dy]
    xyi = np.floor(xyi, xyi).T

    # Next, make a 2D histogram of x & y.
    # Exploit a sparse coo_matrix avoiding np.histogram2d due to excessive
    # memory usage with many points
    grid = coo_matrix((weights, xyi), shape=(nx, ny)).toarray()

    # Kernel Preliminary Calculations ---------------------------------------
    # Calculate the covariance matrix (in pixel coords)
    cov = np.cov(xyi)

    if nocorrelation:
        cov[1,0] = 0
        cov[0,1] = 0

    # Scaling factor for bandwidth
    scotts_factor = n ** (-1.0 / 6.) * adjust  # For 2D

    # Make the gaussian kernel ---------------------------------------------

    # First, determine the bandwidth using Scott's rule
    # (note that Silvermann's rule gives the # same value for 2d datasets)
    std_devs = np.sqrt(np.diag(cov))
    kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)

    # Determine the bandwidth to use for the gaussian kernel
    inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

    # x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
    xx = np.arange(kern_nx, dtype=np.float) - kern_nx / 2.0
    yy = np.arange(kern_ny, dtype=np.float) - kern_ny / 2.0
    xx, yy = np.meshgrid(xx, yy)

    # Then evaluate the gaussian function on the kernel grid
    kernel = np.vstack((xx.flatten(), yy.flatten()))
    kernel = np.dot(inv_cov, kernel) * kernel
    kernel = np.sum(kernel, axis=0) / 2.0
    kernel = np.exp(-kernel)
    kernel = kernel.reshape((kern_ny, kern_nx))

    # Convolve the histogram with the gaussian kernel
    # use boundary=symm to correct for data boundaries in the kde
    grid = convolve2d(grid, kernel, mode='same', boundary='symm')

    # Normalization factor to divide result by so that units are in the same
    # units as scipy.stats.kde.gaussian_kde's output.
    norm_factor = 2 * np.pi * cov * scotts_factor ** 2
    norm_factor = np.linalg.det(norm_factor)
    norm_factor = n * dx * dy * np.sqrt(norm_factor)

    # Normalize the result
    grid /= norm_factor

    return grid, (xmin, xmax, ymin, ymax), dx, dy


def main(fname='Galaxy_NSCs_reff_early.dat', nsamp=50, nboot=1000,
         savefig=False, xfloor=10, yfloor=10, xnorm=6, ynorm=0.5,
         null_threshold=1e-6, sigma_samp=0, x12label='', y12label='',
         extents=None, autonorm=True):
    """ Main function that runs the fit and plots

        Parameters
        ----------

        fname: str
            file to get the data from
            both 4 or 6 columns will work. 4 columns assumes x-uncertainties
            only.

        nsamp: int, optional
            number of samples to represent per data point uncertainties

        nboot: int, optional
            number of bootstrap realization

        xnorm: float, optional
            x normalization of the fit (data normalization)

        ynorm: float, optional
            y normalization of the fit (data normalization)

        xfloor: float, optional
            x-value uncertainty (in %) if there is no measurement error

        yfloor: float, optional
            y-value uncertainty (in %) if there is no measurement error

        null_threshold: float, optional
            threadhod for considering null uncertainties

        sigma_samp: int, optional
            if > 0, the plot will include model representations of sigma_samp
            draws over the intrinsic dispersion given a slope and intercept

        extents: tuple
            optional (xmin, xmax, ymin, ymax) extensions of the main plot

        autonorm: bool
            set to find the autonormalization of the data
    """

    # read the file
    data = np.loadtxt(fname)

    if data.shape[1] == 4:
        # only uncertainties on x-axis
        x, y, xmin, xmax = data.T
        ymin = ymax = np.zeros(len(x))
    elif data.shape[1] == 6:
        # uncertainties on x-axis and y-axis
        x, y, xmin, xmax, ymin, ymax = data.T
    else:
        raise AttributeError('File does not contain a format that I expected')

    ymin += (ymin < null_threshold) * (yfloor / 100.) * y
    ymax += (ymax < null_threshold) * (yfloor / 100.) * y
    xmin += (xmin < null_threshold) * (xfloor / 100.) * x
    xmax += (xmax < null_threshold) * (xfloor / 100.) * x

    # prepare figure
    # the figure is created on the fly to optimize a bit
    plt.figure()
    ax_xy = plt.subplot(222)

    # bootstrap: draw uniformly N points from N data points.
    # plot at the same time to optimize a bit
    models = []
    logxmod = np.linspace(np.log10(x.min()), np.log10(x.max()), 300)
    ylim = [np.log10(y.min()), np.log10(y.max())]

    # Trying to optimize plot transparencies
    plt_alpha = 0.8 / (5 * np.log10(nboot * nsamp * 100 * (1 + sigma_samp)))

    if autonorm is True:
        xnorm = np.nanmean(np.log10(x))
        ynorm = np.nanmean(np.log10(y))
        print('Autonorms: ', xnorm, ynorm)

    for k in range(nboot):
        ind = np.random.randint(0, len(x), len(x))
        alpha, beta, delta = MAP(x[ind], y[ind],
                                 xmin[ind], xmax[ind],
                                 ymin[ind], ymax[ind],
                                 nsamp=nsamp,
                                 xnorm=xnorm, ynorm=ynorm)
        models.append((alpha, beta, delta))
        logymod = (alpha * (logxmod - xnorm) + beta + ynorm)
        ylim[0] = min(ylim[0], logymod.min())
        ylim[1] = max(ylim[1], logymod.max())
        ax_xy.loglog(10 ** logxmod, 10 ** logymod, color='b',
                     alpha=plt_alpha, zorder=-100)
        if sigma_samp > 0:
            # add dispersion
            v = 1. / np.sqrt( 1. + alpha ** 2) * np.array([-alpha, 1 ])
            for _ in range(sigma_samp):
                offset = np.random.normal(0, delta, 1)
                ax_xy.loglog(10 ** (logxmod + v[0] * offset),
                             10 ** (logymod + v[1] * offset),
                             color='b', zorder=-10, alpha=plt_alpha)

    # make some stats on the MAP outputs
    models = np.array(models)

    # find the most probable value
    im, e, dx, dy = fastkde(models[:, 0], models[:, 1], gridsize=(100, 100))
    best_idx = (im.argmax() / im.shape[1], im.argmax() % im.shape[1])
    best = (best_idx[0] * dx + e[0], best_idx[1] * dy + e[2])
    alpha, beta = best
    print(" Mode (Alpha, Beta): {0:0.3f}, {1:0.3f}".format(*best))
    # print some stats
    CI_vals = 0.5
    print("    CI: {0}".format(CI_vals))
    alpha_hpd = hpd(models[:, 0], CI_vals)
    beta_hpd = hpd(models[:, 1], CI_vals)
    sigma_hpd = hpd(models[:, 2], CI_vals)
    print(" Alpha: {1:0.3f}  {0}".format(alpha_hpd, models[:, 0].mean()))
    print("  Beta: {1:0.3f}  {0}".format(beta_hpd, models[:, 1].mean()))
    print(" Sigma: {1:0.3f}  {0}".format(sigma_hpd, models[:, 2].mean()))

    # plot data and uncertainties
    _, _, delta = 3 * np.median(models, axis=0) - 2 * models.mean(0)
    logymod = (alpha * (logxmod - xnorm) + beta + ynorm)
    v = 1. / np.sqrt( 1. + alpha ** 2) * np.array([-alpha, 1 ])
    ax_xy.loglog(10 ** (logxmod + v[0] * delta),
                 10 ** (logymod + v[1] * delta),
                 'r-', zorder=-10, alpha=0.5)
    ax_xy.loglog(10 ** (logxmod - v[0] * delta),
                 10 ** (logymod - v[1] * delta),
                 'r-', zorder=-10, alpha=0.5)
    ax_xy.loglog(10 ** logxmod, 10 ** logymod, 'r-', lw=2, zorder=-10)
    ax_xy.loglog(x, y, 'k.')
    ax_xy.errorbar(x, y, xerr=[xmin,xmax], yerr=[ymin, ymax], linestyle='none', color='k')

    # polish limits
    if extents is None:
        margins = 0.05  # fractional margins
        xlim = (logxmod.min(), logxmod.max())
        xlim = [xlim[0] - margins * np.diff(xlim), xlim[1] + margins * np.diff(xlim)]
        ax_xy.set_xlim(10 ** xlim[0], 10 ** xlim[1])
        ylim = [ylim[0] - margins * np.diff(ylim), ylim[1] + margins * np.diff(ylim)]
        ax_xy.set_ylim(10 ** ylim[0], 10 ** ylim[1])
    else:
        if type(extents) in basestring:
            extents = [float(k) for k in extents.split(',')]
        ax_xy.set_xlim(10 ** extents[0], 10 ** extents[1])
        ax_xy.set_ylim(10 ** extents[2], 10 ** extents[3])

    _xlabel = r'' + x12label
    _ylabel = r'' + y12label
    ax_xy.set_xlabel(_xlabel)
    ax_xy.set_ylabel(_ylabel)

    text = r'$\log_{{10}}(\frac{{\text{{' + y12label + r'}}}}'
    text += r'{{{ynorm:.3g}}}) = \alpha\,\log_{{10}}(\frac{{'.format(ynorm=10 ** ynorm)
    text += r'\text{{' + x12label + '}}'
    text += r'}}{{{xnorm:0.3g}}}) + \beta\\ _.\qquad\qquad\qquad\qquad\qquad\qquad\sigma={sigma:0.3f}$'.format(xnorm=10 ** xnorm, sigma=models[:, 2].mean() )

    txt = ax_xy.text(0.97, 0.01, text,
                     transform=ax_xy.transAxes, horizontalalignment='right')
    # txt.set_path_effects([PathEffects.Stroke(linewidth=3, foreground="w"),
    #                       PathEffects.Normal()])

    ax1 = plt.subplot(221)
    n, _, _ = ax1.hist(models[:, 0], 30, alpha=0.5, histtype='stepfilled',
                       color='k')
    ax1.vlines(best[0], 0, n.max(), color='r')
    txt = ax1.text(0.95, 0.85, r'$\alpha = {0:0.3f}^{{+{1:0.3f} }}_{{-{2:0.3f}}}$'.format(best[0], alpha_hpd[1] - best[0], best[0] - alpha_hpd[0]), transform=ax1.transAxes, horizontalalignment='right')
    txt.set_path_effects([PathEffects.Stroke(linewidth=3, foreground="w"),
                          PathEffects.Normal()])

    ax1.set_ylabel('N')

    ax2 = plt.subplot(224)
    n, _, _ = ax2.hist(models[:, 1], 30, alpha=0.5, orientation='horizontal',
                       histtype='stepfilled', color='k')
    txt = ax2.text(0.95, 0.1, r'$\beta = {0:0.3f}^{{+{1:0.3f} }}_{{-{2:0.3f}}}$'.format(best[1], beta_hpd[1] - best[1], best[1] - beta_hpd[0]), transform=ax2.transAxes, horizontalalignment='right')
    txt.set_path_effects([PathEffects.Stroke(linewidth=3, foreground="w"),
                          PathEffects.Normal()])

    ax2.hlines(best[1], 0, n.max(), color='r')
    ax2.set_xlabel('N')

    ax3 = plt.subplot(223, sharex=ax1, sharey=ax2)

    ax3.contour(im.T, extent=e, cmap=plt.cm.Blues_r, zorder=1000)
    ax3.plot(models[:, 0], models[:, 1], 'k.', alpha=0.4)
    ax3.vlines(best[0], *ax3.get_ylim(), color='r')
    ax3.hlines(best[1], *ax3.get_xlim(), color='r')
    ax3.set_xlabel(r'$\alpha$')
    ax3.set_ylabel(r'$\beta$')
    ax3.set_xlim(models[:, 0].min(), models[:, 0].max())
    ax3.set_ylim(models[:, 1].min(), models[:, 1].max())

    # polish the figure
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    if figrc is not None:
        # figrc.hide_axis('bottom left'.split(), ax=ax_xy)
        figrc.setNmajors(xval=5, yval=5, ax=ax3)
        # figrc.hide_axis('top right left'.split(), ax=ax1)
        figrc.setNmajors(xval=5, ax=ax1)
        # figrc.hide_axis('top right bottom'.split(), ax=ax2)
        figrc.setNmajors(yval=5, ax=ax2)
        figrc.setMargins(wspace=0.05, hspace=0.05)
        ax_xy.yaxis.set_label_position("right")
        ax_xy.xaxis.set_label_position("top")
        ax_xy.xaxis.set_ticks_position("top")
        ax_xy.yaxis.set_ticks_position("right")
    else:
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if savefig not in (False, 'False', 'false'):
        plt.savefig('{outname:s}.{ext:s}'.format(outname=fname.split('.')[0],
                                                 ext=savefig ))
    plt.show()

    return models


# declare what to do from a command line call
# and only from a command line call
if __name__ == '__main__':

    # read config file first to get default values
    import pyconfig
    import os

    opts = dict(
        default=(
            ('-n', '--nsamp',   dict(dest="nsamp", help="number of samples to represent per data point uncertainties", default=20, type='int')),
            ('-N', '--nboot',   dict(dest="nboot", help="number of bootstrap realization", default=100, type='int')),
            ('-c', '--config',   dict(dest="configfname", help="Configuration file to use for default values", default='None', type='str')),
            ('--xnorm',   dict(dest="xnorm", help="x-data normalization value", default=6, type='float')),
            ('--ynorm',   dict(dest="ynorm", help="y-data normalization value", default=0.5, type='float')),
            ('--autonorm', dict(action="store_true", default=False, dest='autonorm', help='Auto normalization values')),
            ('--xfloor',   dict(dest="xfloor", help="floor of x-value uncertainty (in %)", default=10, type='float')),
            ('--yfloor',   dict(dest="yfloor", help="floor of y-value uncertainty (in %)", default=10, type='float')),
        ),
        outputs=(
            ('-o', '--output',   dict(dest="output", help="export the samples into a file", default='None', type='str')),
            ('-f', '--savefig', dict(dest="savefig", default='False',  help="Generate figures with the desired format (pdf, png...)", type='str')),
        ),
        plotting=(
            ('--x12label',   dict(dest="x12label", help="X-label of the top-right plot (it can be in latex form)", default='None', type='str')),
            ('--y12label',   dict(dest="y12label", help="Y-label of the top-right plot (it can be in latex form)", default='None', type='str')),
            ('--sigma_samp',   dict(dest="sigma_samp", help="number of samplings to represent the intrinsic dispersion of the plot", default=0, type='int')),
            ('--extents',   dict(dest="extents", help="xmin, xmax, ymin, ymax values of the main plot (log values)", default=None, type='str')),
        )
    )

    parser = pyconfig.make_parser_from_options(*opts.items(), usage=__doc__)

    (options, args) = parser.parse_args()

    configfname = options.__dict__.pop('configfname')

    if configfname not in [None, 'None', 'none']:
        if os.path.isfile(configfname):
            parser = pyconfig.update_default_from_config_file(configfname,
                                                              *opts.items(),
                                                              usage=__doc__)
        else:
            # save configuration
            print('Exporting configuration to {0}'.format(configfname))
            txt = pyconfig.generate_conf_file_from_opts(*opts.items())
            with open(configfname, 'w') as f:
                f.write(txt)

    (options, args) = parser.parse_args()

    configfname = options.__dict__.pop('configfname')
    output = options.__dict__.pop('output')

    models = main(*args, **options.__dict__)

    if output not in ('None', 'none', None):
        np.savetxt(output, models, header='slope intercept dispersion')
