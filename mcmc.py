import numpy as np
import pylab as plt
import figrc
import emcee
from pbar import Pbar
import signal
import functools
from multiprocessing.pool import Pool
from multiprocessing import TimeoutError
from scipy.sparse import coo_matrix
from scipy.signal import convolve2d

figrc.ezrc(12, 1, 16, 5, 2)


def resample_data_pdf(x, y, xmin, xmax, ymin, ymax, nsamp=1000, xnorm=1.,
                      ynorm=1., errorfloor=1e-4):
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
    xdata: ndarray, dtype=float
        sample of x values

    ydata: ndarray, dtype=float
        sample of y values
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

    xdata = np.array(xdata).ravel()
    ydata = np.array(ydata).ravel()
    ind = np.isfinite(xdata) & np.isfinite(ydata) & (xdata > 0) & (ydata > 0)
    xdata = xdata[ind]
    ydata = ydata[ind]

    return xdata / xnorm, ydata / ynorm


def lsq_orthogonal_to_line(xdata, ydata, alpha, beta, sigma):
    """ Project (x,y) into the line reference system

    Parameters
    ----------
    xdata: ndarray, dtype=float
        sample of x values

    ydata: ndarray, dtype=float
        sample of y values

    alpha: float
        slope of the relation

    beta: float
        intercept of the relation

    sigma: float
        intrinsic dispersion

    returns
    -------
    delta: float
        sum of lsq distances between the data and the line
    """
    # trying to get residual perpendicularly to the line
    v = 1. / np.sqrt( 1. + alpha ** 2) * np.array([-alpha, 1 ])
    # add intrinsic dispersion
    v /= sigma ** 2
    vects = np.vstack([xdata - xdata.mean(), ydata - ydata.mean()]).T
    off = beta * np.cos(np.arctan(alpha))
    delta = (np.array([np.dot(v, k) - off for k in vects]) ** 2).sum()
    return delta / (2. * sigma ** 2)


def to_log10(xdata, ydata):
    """ convert to log values with proper filters """
    ind = np.isfinite(xdata) & np.isfinite(ydata) & (xdata > 0) & (ydata > 0)
    return (np.log10(xdata[ind]), np.log10(ydata[ind]))


def loadtxt(fname, xfloor=10, yfloor=10, null_threshold=1e-6,):
    """ Load a file and filter for proper formatting

    Parameters
    ----------

    fname: str
        file to get the data from
        both 4 or 6 columns will work. 4 columns assumes x-uncertainties
        only.

    xfloor: float, optional
        x-value uncertainty (in %) if there is no measurement error

    yfloor: float, optional
        y-value uncertainty (in %) if there is no measurement error

    null_threshold: float, optional
        threadhod for considering null uncertainties

    returns
    -------

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

    return (x, y, xmin, xmax, ymin, ymax)


# Create some convenience routines for plotting =================

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, e, dx, dy, xbins, ybins = fastkde(trace1, trace2, gridsize=(nbins, nbins))

    best_idx = (L.argmax() / L.shape[1], L.argmax() % L.shape[1])
    best = (best_idx[0] * dx + e[0], best_idx[1] * dy + e[2])

    L[L == 0] = 1E-16

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    return L_cumsum[i_unsort].reshape(shape), e, best


def plot_MCMC_trace(ax, ax_alpha, ax_beta, xdata, ydata, trace, scatter=True,
                    **kwargs):
    """Plot traces and contours"""
    sigma, e, best = compute_sigma_level(trace[0], trace[1], nbins=50)
    ax.contour(sigma.T, extent=e, levels=[0.683, 0.955, 0.997],
               cmap=plt.cm.Oranges)

    ax.contour(sigma.T, extent=e, levels=[0.01, 0.05, 0.10, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               cmap=plt.cm.Greys)
    if scatter:
        ax.plot(trace[0], trace[1], '.k', alpha=0.1)
    ax.set_xlabel(r'slope')
    ax.set_ylabel(r'intercept')

    n, _, _ = ax_alpha.hist(trace[0], 30, alpha=0.5, histtype='stepfilled',
                            color='k')
    n, _, _ = ax_beta.hist(trace[1], 30, alpha=0.5, orientation='horizontal',
                           histtype='stepfilled', color='k')

    ax_alpha.set_ylabel('N')
    ax_beta.set_xlabel('N')

    print(" Mode (Alpha, Beta): {0:0.3f}, {1:0.3f}".format(*best))
    CI_vals = 0.5
    print("    CI: {0}".format(CI_vals))
    alpha_hpd = hpd(trace[0], CI_vals)
    beta_hpd = hpd(trace[1], CI_vals)
    sigma_hpd = hpd(trace[2], CI_vals)
    print(" Alpha: {1:0.3f}  {0}".format(alpha_hpd, trace[0].mean()))
    print("  Beta: {1:0.3f}  {0}".format(beta_hpd, trace[1].mean()))
    print(" Sigma: {1:0.3f}  {0}".format(sigma_hpd, trace[2].mean()))
    return best, alpha_hpd, beta_hpd, sigma_hpd


def plot_MCMC_model(ax, x, y, xmin, xmax, ymin, ymax, trace,
                    loglog=True, xlabel='x', ylabel='y', xnorm=0, ynorm=0):
    """Plot the linear model and 1-sigma contours"""
    alpha, beta, sigma = trace[:3]

    margins = 0.05  # fractional margins

    if loglog is True:
        Npts = 100
        xlog = np.log10(x)
        ylog = np.log10(y)
        xfit = np.linspace(xlog.min(), xlog.max(), Npts)

        # make limits
        xlim = (xlog.min(), xlog.max())
        xlim = [xlim[0] - margins * np.diff(xlim), xlim[1] + margins * np.diff(xlim)]
        ylim = (ylog.min(), ylog.max())
        ylim = [ylim[0] - margins * np.diff(ylim), ylim[1] + margins * np.diff(ylim)]

        # plot data
        ax.loglog(x, y, 'k.', zorder=100)
        ax.errorbar(x, y, xerr=[xmin,xmax], yerr=[ymin,ymax], linestyle='none',
                    color='k', zorder=100)

        # plot model
        xfit = np.linspace(xlog.min(), xlog.max(), Npts)
        # off = sigma * np.cos(np.arctan(alpha))
        yfit = alpha[:, None] * (xfit - xlog.mean()) + beta[:, None] + ylog.mean()
        # yfit = alpha[:, None] * (xfit - xnorm) + beta[:, None] + ynorm
        mu = yfit.mean(0)
        sig1 = np.percentile(yfit, [16, 84], axis=0)
        sig2 = np.percentile(yfit, [2.5, 97.5], axis=0)
        sig3 = np.percentile(yfit, [0.1, 99.9], axis=0)

        ax.fill_between(10 ** xfit, 10 ** sig3[0], 10 ** sig3[1], color='lightgray', alpha=0.2)
        ax.fill_between(10 ** xfit, 10 ** sig2[0], 10 ** sig2[1], color='lightgray', alpha=0.4)
        ax.fill_between(10 ** xfit, 10 ** sig1[0], 10 ** sig1[1], color='lightgray', alpha=0.6)
        ax.plot(10 ** xfit, 10 ** mu, '-k')

        ax.set_xlim(10 ** xlim[0], 10 ** xlim[1])
        ax.set_ylim(10 ** ylim[0], 10 ** ylim[1])
    else:
        Npts = 10

        # make limits
        xlim = (x.min(), x.max())
        xlim = [xlim[0] - margins * np.diff(xlim), xlim[1] + margins * np.diff(xlim)]
        ylim = (y.min(), y.max())
        ylim = [ylim[0] - margins * np.diff(ylim), ylim[1] + margins * np.diff(ylim)]

        ax.plot(x, y, 'k.', zorder=100)
        ax.errorbar(x, y, xerr=[xmin,xmax], yerr=[ymin,ymax], linestyle='none',
                    color='k', zorder=100)

        xfit = np.linspace(x.min(), x.max(), Npts)
        # off = sigma * np.cos(np.arctan(alpha))
        yfit = alpha[:, None] * (xfit - x.mean()) + beta[:, None] + y.mean()
        mu = yfit.mean(0)
        sig = np.percentile(yfit, [16, 84], axis=0)

        # ind = min(100, len(yfit))
        # ax.plot(xfit, yfit[-ind:].T, color='b', alpha=0.01)
        ax.fill_between(xfit, sig[0], sig[1], color='lightgray')
        ax.plot(xfit, mu, '-k')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_MCMC_results(x, y, xmin, xmax, ymin, ymax, trace, colors='k',
                      xlabel='x', ylabel='y', xnorm=0, ynorm=0):
    """Plot both the trace and the model together"""
    plt.figure()

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(224)
    ax3 = plt.subplot(223, sharex=ax1, sharey=ax2)
    ax_xy = plt.subplot(222)
    best, alpha_hpd, beta_hpd, sigma_hpd = plot_MCMC_trace(ax3, ax1, ax2, x, y,
                                                           trace, scatter=True)
    plot_MCMC_model(ax_xy, x, y, xmin, xmax, ymin, ymax, trace, xlabel=xlabel,
                    ylabel=ylabel, xnorm=0, ynorm=0)

    ax1.vlines(best[0], 0, ax1.get_ylim()[1], color='r')
    ax2.hlines(best[1], 0, ax2.get_xlim()[1], color='r')
    ax3.vlines(best[0], *ax3.get_ylim(), color='r')
    ax3.hlines(best[1], *ax3.get_xlim(), color='r')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
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

    return best, alpha_hpd, beta_hpd, sigma_hpd


class LnPosterior(object):
    """ keep it tight and precompute everything possible

    Attributes
    ----------
    fname: str
        file to get the data from
        both 4 or 6 columns will work. 4 columns assumes x-uncertainties
        only.

    nsamp: int, optional
        number of samples to represent per data point uncertainties

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
    """

    def __init__(self, fname, nsamp=20, errorfloor=1e-4,
                 xfloor=10, yfloor=10, xnorm=1, ynorm=1., null_threshold=1e-6):
        self.fname = fname
        self.inputs = loadtxt(fname, xfloor, yfloor, null_threshold)
        self.norms = (xnorm, ynorm)
        self.errorfloor = errorfloor
        self.nsamp = nsamp
        self.null_threshold = null_threshold
        self.ndim = 3

        self.reset()

    def __len__(self):
        return self.ndim

    def reset(self, bootstrap=0, **kwargs):
        """ init fake data """
        kw = dict(nsamp=self.nsamp, xnorm=self.norms[0], ynorm=self.norms[1],
                  errorfloor=self.errorfloor)
        kw.update(**kwargs)
        if (bootstrap <= 0):
            self.data = to_log10(*resample_data_pdf(*self.inputs, **kw))
        else:
            N = len(self.inputs[0])
            ind = np.random.randint(0, N,  N)
            inputs = [k[ind] for k in self.inputs]
            self.data = to_log10(*resample_data_pdf(*inputs, **kw))

    def lnlike(self, theta):
        """
        transform the data into log10 values and compute the log-likelihood

        Parameters
        ----------
        theta: sequence
            parameter vector (alpha, beta, sigma)

        Returns
        -------
        lnlike:float
            log-likelihood of the data with N(y = alpha x + beta; sigma)
        """
        alpha, beta, sigma = theta
        xlog, ylog = self.data
        lnp = lsq_orthogonal_to_line(xlog, ylog, alpha, beta, sigma)
        lnp = -0.5 * len(xlog) * np.log(2 * np.pi * sigma ** 2) - 0.5 * lnp
        return lnp

    def lnprior(self, theta):
        """
        Prior on Slope and Intercept
        ============================

        If our model is given by :math:`y= \beta + \alpha x`

        then we can construct a parameter-space probability element
        :math:`p(\beta, \alpha)d\beta d\alpha`

        Because x and y are symmetric, we could just as easily use another set
        of parameters:
        :math:`x= \beta' y + \alpha'`

        with probability element :math:`Q(\beta',\alpha') d\beta'd\alpha'`,
        where it's easy to show that in this case,

        :math:`(\beta', \alpha')=(-\alpha^{-1} \beta, \alpha^{-1})`

        From the Jacobian of the transformation, we can show that

        Q(\beta',\alpha')=\alpha^3 p(\beta,\alpha).

        Maintaining the symmetry of the problem requires that this change of
        variables should not affect the prior probability, so we can write:

        \alpha^3 p(\beta,\alpha) = p(-\alpha^{-1} \beta,\alpha^{-1})

        This is a functional equation which is satisfied by

        p(\beta,\alpha) ~ (1 + \alpha^2)^{-3/2}.

        which is equivalent to saying that `\beta` is uniformly distributed, and
        `\alpha` is distributed uniformly in `sin\theta` where
        `\theta =\tan^{−1}(\alpha).

        This might surprise you that the slopes are distributed according to
        `sin(\theta)` rather than uniformly in angle `\theta`. This term,
        though, can actually be thought of as coming from the intercept itself.

        If we change variables to the orthogonal reference system, then it's
        straightforward to show that our variables are uniformly distributed.

        Prior on sigma
        ==============

        Similarly, we want the prior on :math:`\sigma`, the intrinsic orthogonal
        dispersion to the relation, to be invariant to rescalings of the problem
        (i.e. changing units). So our probability must satisfy

        p(\sigma)d\sigma=P(\sigma/c)d\sigma/c.

        This is a functional equation satisfied by P(\sigma) ~ 1/\sigma.

        This is known as the Jeffreys Prior, after Harold Jeffreys.

        Altogether
        =========
        P(\alpha,\beta,\sigma) ~ 1/\sgima  (1 + \alpha ^ 2)^{-3/2}
        """
        alpha, beta, sigma = theta
        if sigma < 0:
            return -np.inf
        else:
            return -1.5 * np.log(1 + alpha ** 2) - np.log(sigma)

    def lnposterior(self, theta):
        lnp = self.lnprior(theta)
        if np.isfinite(lnp):
            lnp = lnp + self.lnlike(theta)
        # print(theta, lnp)
        # return self.lnlike(theta)
        return lnp

    def initial_guess(self):
        xdata, ydata = self.data
        cov = np.cov(xdata, ydata)

        alpha = cov[1,0] / (cov[0, 0] ** 2)
        beta = (ydata).mean() - alpha * (xdata).mean()

        # trying to get residual perpendicularly to the line
        v = 1. / np.sqrt( 1. + alpha ** 2) * np.array([-alpha, 1 ])
        vects = np.vstack([xdata - xdata.mean(), ydata - ydata.mean()]).T
        delta = (np.array([np.dot(v, k) for k in vects]) ** 2).sum()

        return alpha, beta, (0.5 * delta / (len(xdata) - 1)) ** 0.5

    def __call__(self, theta, **kwargs):
        """ Make a function/callable object"""
        return self.lnposterior(theta, **kwargs)


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
    std_devs = np.diag(np.sqrt(cov))
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

    return grid, (xmin, xmax, ymin, ymax), dx, dy, xx, yy


def run_mcmc_with_pbar(sampler, pos0, N, rstate0=None, lnprob0=None, desc=None,
                       **kwargs):
    """
    Iterate :func:`sample` for ``N`` iterations and return the result while also
    showing a progress bar

    Parameters
    ----------
    pos0:
        The initial position vector.  Can also be None to resume from
        where :func:``run_mcmc`` left off the last time it executed.

    N:
        The number of steps to run.

    lnprob0: (optional)
        The log posterior probability at position ``p0``. If ``lnprob``
        is not provided, the initial value is calculated.

    rstate0: (optional)
        The state of the random number generator. See the
        :func:`random_state` property for details.

    desc: str (optional)
        title of the progress bar

    kwargs: (optional)
        Other parameters that are directly passed to :func:`sample`.

    Returns
    -------
    t: tuple
        This returns the results of the final sample in whatever form
        :func:`sample` yields.  Usually, that's: ``pos``, ``lnprob``,
        ``rstate``, ``blobs`` (blobs optional)
    """
    if pos0 is None:
        if sampler._last_run_mcmc_result is None:
            raise ValueError("Cannot have pos0=None if run_mcmc has never "
                             "been called.")
        pos0 = sampler._last_run_mcmc_result[0]
        if lnprob0 is None:
            rstate0 = sampler._last_run_mcmc_result[1]
        if rstate0 is None:
            rstate0 = sampler._last_run_mcmc_result[2]

    with Pbar(maxval=N, desc=desc) as pb:
        k = 0
        for results in sampler.sample(pos0, lnprob0, rstate0, iterations=N,
                                      **kwargs):
            k += 1
            pb.update(k)

    # store so that the ``pos0=None`` case will work.  We throw out the blob
    # if it's there because we don't need it
    sampler._last_run_mcmc_result = results[:3]

    return results


def _initializer_wrapper(actual_initializer, *rest):
    """
    We ignore SIGINT. It's up to our parent to kill us in the typical
    condition of this arising from ``^C`` on a terminal. If someone is
    manually killing us with that signal, well... nothing will happen.

    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if actual_initializer is not None:
        actual_initializer(*rest)


class InterruptiblePool(Pool):
    """
    A modified version of :class:`multiprocessing.pool.Pool` that has better
    behavior with regard to ``KeyboardInterrupts`` in the :func:`map` method.

    This pool can also be used as a contextmanager (`with`)

    Parameters
    ----------
    processes: int, optional
        The number of worker processes to use; defaults to the number of CPUs.

    initializer: callable, optional
        Either ``None``, or a callable that will be invoked by each worker
        process when it starts.

    initargs: sequence, optional
        Arguments for *initializer*; it will be called as
        ``initializer(*initargs)``.

    kwargs: dict, optional
        Extra arguments. Python 2.7 supports a ``maxtasksperchild`` parameter.

    """
    wait_timeout = 3600

    def __init__(self, processes=None, initializer=None, initargs=(),
                 **kwargs):
        if (processes != 0):
            new_initializer = functools.partial(_initializer_wrapper, initializer)
            super(InterruptiblePool, self).__init__(processes, new_initializer,
                                                    initargs, **kwargs)
        else:
            self._processes = 0

    def map(self, func, iterable, chunksize=None):
        """
        Equivalent of ``map()`` built-in, without swallowing
        ``KeyboardInterrupt``.

        :param func:
            The function to apply to the items.

        :param iterable:
            An iterable of items that will have `func` applied to them.

        """
        if self._processes != 0:
            # The key magic is that we must call r.get() with a timeout, because
            # a Condition.wait() without a timeout swallows KeyboardInterrupts.
            r = self.map_async(func, iterable, chunksize)

            while True:
                try:
                    return r.get(self.wait_timeout)
                except TimeoutError:
                    pass
                except KeyboardInterrupt:
                    self.terminate()
                    self.join()
                    raise
                # Other exceptions propagate up.
        else:
            return map(func, iterable)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self._processes != 0:
            self.terminate()
            self.join()
        return False


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
    tx = np.transpose(x, list(range(x.ndim)[1:]) + [0])
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


def main(fname='Galaxy_NSCs_reff_late.dat', xlabel='x', ylabel='y', nsamp=1,
         nboot=10, xnorm=9.294, ynorm=0.687, nwalkers=50, nburn=20, nsteps=30,
         threads=4, savefig=False, errorfloor=1e-4, xfloor=10, yfloor=10,
         null_threshold=1e-6):
    """ run the fit

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

    xlabel: str
     X-label of the top-right plot (it can be in latex form)

    ylabel: str
     Y-label of the top-right plot (it can be in latex form)
    """

    lnposterior = LnPosterior(fname, xnorm=xnorm, ynorm=ynorm, nsamp=nsamp,
                              xfloor=xfloor, yfloor=yfloor,
                              null_threshold=null_threshold,
                              errorfloor=errorfloor)

    # number of parameters in the model
    ndim = lnposterior.ndim

    np.random.seed(0)

    with InterruptiblePool(threads) as pool:
        if nboot > 0:
            emcee_trace = []
            for k in range(nboot):
                lnposterior.reset(bootstrap=1)   # resample data
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior,
                                                pool=pool)
                starting_guesses = np.random.normal(0.5, 0.001, (nwalkers, ndim))
                print('')
                run_mcmc_with_pbar(sampler, starting_guesses, nsteps,
                                   desc='sampling {0:d}'.format(k + 1))
                emcee_trace.append(sampler.chain[:, nburn:, :].reshape(-1, ndim).T)
            emcee_trace = np.hstack(emcee_trace)
        else:
            lnposterior.reset()   # resample data
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior,
                                            pool=pool)
            starting_guesses = np.random.normal(0.5, 0.1, (nwalkers, ndim))
            run_mcmc_with_pbar(sampler, starting_guesses, nsteps, desc='sampling')
            emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T

    # sampler.chain is of shape (nwalkers, nsteps, ndim)
    # we'll throw-out the burn-in points and reshape:
    xlog, ylog = lnposterior.data

    plot_MCMC_results(*(tuple(lnposterior.inputs) + (emcee_trace,)),
                      xlabel=xlabel, ylabel=ylabel, xnorm=xnorm, ynorm=ynorm)

    if savefig not in (False, 'False', 'false'):
        plt.savefig('{outname:s}.{ext:s}'.format(outname=fname.split('.')[0],
                                                 ext=savefig ))

    return emcee_trace.T


if __name__ == '__main__':

    opts = (
        ('-f', '--savefig', dict(dest="savefig", default='False',  help="Generate figures with the desired format (pdf, png...)", type='str')),
        ('-n', '--nsamp',   dict(dest="nsamp", help="number of samples to represent per data point uncertainties", default=1, type='int')),
        ('-N', '--nboot',   dict(dest="nboot", help="number of bootstrap realization", default=10, type='int')),
        ('--xnorm',   dict(dest="xnorm", help="x-data normalization value", default=6, type='float')),
        ('--ynorm',   dict(dest="ynorm", help="y-data normalization value", default=0.5, type='float')),
        ('--xfloor',   dict(dest="xfloor", help="floor of x-value uncertainty (in %)", default=10, type='float')),
        ('--yfloor',   dict(dest="yfloor", help="floor of y-value uncertainty (in %)", default=10, type='float')),
        ('--x12label',   dict(dest="xlabel", help="X-label of the top-right plot (it can be in latex form)", default='None', type='str')),
        ('--y12label',   dict(dest="ylabel", help="Y-label of the top-right plot (it can be in latex form)", default='None', type='str')),
        ('-o', '--output',   dict(dest="output", help="export the samples into a file", default='None', type='str')),
        ('--nwalkers',   dict(dest="nwalkers", help="number of walkers in emcee", default=50, type='int')),
        ('--nburn',   dict(dest="nburn", help="number of steps burned in emcee", default=20, type='int')),
        ('--nsteps',   dict(dest="nsteps", help="number of steps incl. burnining phase in emcee", default=40, type='int')),
        ('--threads',   dict(dest="threads", help="number of parallel threads in emcee", default=4, type='int')),
    )

    from optparse import OptionParser
    parser = OptionParser()

    for ko in opts:
        parser.add_option(*ko[:-1], **ko[-1])

    (options, args) = parser.parse_args()

    output = options.__dict__.pop('output')

    models = main(*args, **options.__dict__)

    if output not in ('None', 'none', None):
        output = '{outname:s}_samples.{ext:s}'.format(outname=args[0].split('.')[0], ext=output)
        np.savetxt(output, models, header='slope intercept dispersion')

    plt.show()
