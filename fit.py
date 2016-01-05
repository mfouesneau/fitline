#!/usr/bin/env python
""" Fitting a straight line with non-symmetric (but Gaussian) uncertainties """
from __future__ import print_function
import numpy as np
# from matplotlib import use
# use('Agg')
import pylab as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


def clean_data(d, null_threshold=1e-6, xfloor=10., yfloor=10.):
    """
    Updates inplace the dataset to replace bad uncertainty values
    with less ridiculous ones

    parameters
    ----------

    d: ndarray (n, >6)
        dataset of n entries
    null_threshold:
        threshold under which uncertainty values are unreliable
    xfloor: float
        percentage fraction of x
    yfloor: float
        percentage fraction of y
    """

    x    = d[:, 0]
    y    = d[:, 1]
    xmin = d[:, 2]
    xmax = d[:, 3]
    ymin = d[:, 4]
    ymax = d[:, 5]

    # update values accordingly
    ind = ymin < null_threshold
    d[ind, 4] += yfloor * 0.01 * y[ind]
    ind = ymax < null_threshold
    d[ind, 5] += yfloor * 0.01 * y[ind]
    ind = xmin < null_threshold
    d[ind, 2] += xfloor * 0.01 * x[ind]
    ind = xmax < null_threshold
    d[ind, 3] += xfloor * 0.01 * x[ind]


def sample_data_pdf(d, rnd, errorfloor=1e-4, bootstrap=False):
    """sample_data_pdf

    parameters
    ----------
    d: ndarray (n, >6)
        data with uncertainties
    rnd: RandomState
        random rnd
    errorfloor: float
        minimum uncertainty to assure in data units
    bootstrap: bool
        set to bootstrap the data

    returns
    -------
    xdata: ndarray (n, )
        created x samples
    ydata: ndarray (n, )
        created y samples
    """

    # select points either all or randomly
    if (bootstrap):
        index = rnd.randint(0, len(d) - 1, size=len(d))
    else:
        index = np.arange(len(d), dtype=int)

    xdata = np.empty(len(d), dtype=float)
    ydata = np.empty(len(d), dtype=float)

    for i, ind in enumerate(index):
        x = d[ind][0]
        y = d[ind][1]
        xmin = d[ind][2]
        xmax = d[ind][3]
        ymin = d[ind][4]
        ymax = d[ind][5]

        # sample y
        if (rnd.rand() > 0.5):
            ydata[i] = y + abs(rnd.randn()) * (ymax + errorfloor)
        else:
            ydata[i] = y - abs(rnd.randn()) * (ymin + errorfloor)

        # sample x
        if (rnd.rand() > 0.5):
            xdata[i] = x + abs(rnd.randn()) * (xmax + errorfloor)
        else:
            xdata[i] = x - abs(rnd.randn()) * (xmin + errorfloor)

    return xdata, ydata


def OLS_MAP(xdata, ydata):
    """OLS_MAP Orthogonal Least-Square Maximum a Posteriori

    parameters
    ----------
    xdata: ndarray (n, 1)
        x samples

    ydata: ndarray (n, 1)
        y samples

    returns
    -------
    theta: tuple(3)
        (slope, intercept, dispersion) of the model
    """
    # cleaning
    ind = np.isfinite(xdata) & np.isfinite(ydata)
    x = xdata[ind]
    y = ydata[ind]
    n = np.sum(ind)

    # precomputing
    mx = x.mean()
    my = y.mean()
    c = np.cov(x, y)
    sxx = n * c[0, 0]
    syy = n * c[1, 1]
    sxy = (n - 1) * c[1, 0]

    # MAP
    a = 0.5 * (syy - sxx + np.sqrt( (syy - sxx) ** 2 + 4 * sxy ** 2) ) / sxy
    b = y.mean() - a * x.mean()

    # trying to get residual perpendicularly to the line
    v = 1. / np.sqrt( 1. + a ** 2) * np.array([-a, 1 ])
    vects = np.vstack([x - mx, y - my]).T
    delta = np.sqrt((np.array([np.dot(v, k) for k in vects]) ** 2).sum())

    return a, b, (delta / float(n - 1)) ** 0.5


def main_data(fname="reff_NSC_Mass_late.dat",
              number_of_samples=2000,
              errorfloor=1e-4,
              bootstrap=True,
              log_xnorm=6.5,
              log_ynorm=1,
              xfloor=10,
              yfloor=10,
              null_threshold=1e-6,
              xlabel='X', ylabel='Y',
              plot=True
              ):
    """main_data

    Parameters
    ----------
    fname: str
        input data file
    number_of_samples: int
        number of samplings in total
    errorfloor: float
        minimum uncertainty to assure in data units
    bootstrap: bool
        set to include bootstrapping of the data
    log_xnorm: float
        normalize data
    log_ynorm: float
        normalize data
    xfloor: float
        fractional error on x in percent for unreliable values
    yfloor: float
        fractional error on x in percent for unreliable values
    null_threshold: float
        threshold under which uncertainties are assumed unreliable
    xlabel: str
        label to put on the X axis
    ylabel: str
        label to put on the Y axis
    plot: bool or str
        set to make the plot. If a string is given, save the figure accordingly

    return
    ------
    theta: ndarray (number_of_samples, 3)
        samples of (slope, intercept, dispersion
    """
    #load and clean the data for bad uncertainties
    print("Loading data")
    mat = np.loadtxt(fname)
    print("  + Datashape = {0:s}".format(str(mat.shape)))
    print("  + Clean data")
    clean_data(mat, null_threshold, xfloor, yfloor)

    print("Performing fit")
    print("  + number of samples from p(D) ", number_of_samples)
    print("  + bootstrap data ", bootstrap)
    print("  + log_norm(x, y) = ", log_xnorm, ",", log_ynorm)
    print("  + errorfloor = ", errorfloor)

    generator = np.random.RandomState(seed=1234)

    theta = np.empty((number_of_samples, 3), dtype=float)

    for i in range(number_of_samples):
        # sample data
        xdata, ydata = sample_data_pdf(mat, generator, errorfloor, bootstrap)
        # transform_data_sample_to_log
        # transform and normalize the data
        ind = (np.isfinite(xdata) & (xdata > 0) &
               (ydata > 0) & np.isfinite(ydata))
        xdata = np.log10(xdata[ind]) - log_xnorm
        ydata = np.log10(ydata[ind]) - log_ynorm
        # get MAP from OLS
        theta[i] = OLS_MAP(xdata, ydata)

    print("Results")
    mean = np.nanmean(theta, axis=0)
    std = np.nanstd(theta, axis=0)
    print("  + a = {0:f} +/- {1:f}".format(mean[0], std[0]))
    print("  + b = {0:f} +/- {1:f}".format(mean[1], std[1]))
    print("  + d = {0:f} +/- {1:f}".format(mean[2], std[2]))

    if not (plot in [False, None, 'False', 'false']):
        print('Plotting')
        # figure
        plt.figure(figsize=(10,6))

        ax = plot_data(*mat.T, fname=fname, xlabel=xlabel, ylabel=ylabel)
        plot_models(mat[:, 0], mat[:, 1], log_xnorm, log_ynorm, theta, ax,
                    xlabel=xlabel, ylabel=ylabel)
        xlim = [np.log10(mat[:, 0].min()), np.log10(mat[:, 0].max())]
        xran = np.ptp(xlim)
        ylim = [np.log10(mat[:, 1].min()), np.log10(mat[:, 1].max())]
        yran = np.ptp(ylim)
        margin = 0.05
        xlim = (xlim[0] - margin * xran, xlim[1] + margin * xran)
        ylim = (ylim[0] - margin * yran, ylim[1] + margin * yran)
        ax.set_xlim(10 ** xlim[0], 10 ** xlim[1])
        ax.set_ylim(10 ** ylim[0], 10 ** ylim[1])
        plot_parameter_samples(theta)
        # if string, unicode etc
        if hasattr(plot, '__iter__') and not (plot in ['true', 'None', 'none', 'True']):
            print('  + saving into {0:s}'.format(plot))
            plt.savefig(plot)
        else:
            print('  + showing figure')
            plt.show()

    return theta


def plot_parameter_samples(theta, f=None):
    from matplotlib.ticker import MaxNLocator

    if f is None:
        f = plt.gcf()

    best = np.median(theta, 0)
    alpha_hpd, beta_hpd, _ = np.array(np.percentile(theta, [16,84], axis=0)).T

    ax_3 = f.add_axes([0.6, 0.5, 0.3, 0.35])
    h, bx, by = np.histogram2d(theta[:, 0], theta[:, 1], 30, normed=True)
    e = bx[0], bx[-1], by[0], by[-1]
    ax_3.contour(h.T, 5, extent=e, cmap=plt.cm.Greys_r)
    ax_3.pcolorfast(bx, by, h.T, cmap=plt.cm.Blues, vmin=h[h > 0].min())
    ax_3.vlines(best[0], e[2], e[3], color='k', zorder=100)
    ax_3.hlines(best[1], e[0], e[1], color='k', zorder=100)
    ax_3.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=(1,2,5,10)))
    ax_3.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=(1,2,5,10)))

    ax1 = f.add_axes([0.6, 0.85, 0.3, 0.12], sharex=ax_3, axis_bgcolor='None')
    n, _, _ = ax1.hist(theta[:, 0], 30, histtype='stepfilled', alpha=0.5,
                       facecolor='#0088fa', edgecolor='k', lw=2)
    ax1.vlines(best[0], 0, n.max(), color='k')
    plt.figtext(0.6, 0.32,
                r'$\alpha = {0:0.3f}^{{+{1:0.3f} }}_{{-{2:0.3f}}}$'.format(best[0], alpha_hpd[1] - best[0], best[0] - alpha_hpd[0]),
                fontsize='x-large')

    ax2 = f.add_axes([0.9, 0.5, 0.1, 0.35], sharey=ax_3, axis_bgcolor='None')
    n, _, _ = ax2.hist(theta[:, 1], 30, orientation='horizontal',
                       histtype='stepfilled', alpha=0.5,
                       facecolor='#0088fa', edgecolor='k', lw=2)
    ax2.hlines(best[1], 0, n.max(), color='k')
    plt.figtext(0.6, 0.25,
                r'$\beta = {0:0.3f}^{{+{1:0.3f} }}_{{-{2:0.3f}}}$'.format(best[1], beta_hpd[1] - best[1], best[1] - beta_hpd[0]),
                fontsize='x-large')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    [sk.set_color('None') for k, sk in ax1.spines.items() if k in 'top left right bottom'.split() ]
    [sk.set_color('None') for k, sk in ax2.spines.items() if k in 'top left right bottom'.split() ]
    [sk.set_color('None') for k, sk in ax_3.spines.items() if k in 'top right'.split() ]
    ax_3.yaxis.set_ticks_position('left')
    ax_3.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    ax1.xaxis.set_ticks_position('none')
    ax2.xaxis.set_ticks_position('none')
    ax_3.xaxis.set_tick_params(direction='out', width=1, size=8)
    ax_3.yaxis.set_tick_params(direction='out', width=1, size=8)
    ax_3.set_xlim(e[0], e[1])
    ax_3.set_ylim(e[2], e[3])
    ax_3.spines['bottom'].set_linewidth(1.)
    ax_3.spines['left'].set_linewidth(1.)

    ax_3.set_xlabel('slope', fontsize='x-large')
    ax_3.set_ylabel('intercept', fontsize='x-large')


def plot_data(x, y, xmin, xmax, ymin, ymax, f=None, fname=None,
              xlabel='X', ylabel='Y'):

    if f is None:
        f = plt.gcf()

    ax = f.add_axes([0.12, 0.2, 0.4, 0.7], axis_bgcolor='w')
    ax.loglog(x, y, 'ko')
    ax.errorbar(x, y, xerr=[xmin,xmax], yerr=[ymin, ymax], linestyle='none',
                color='k')
    ax.xaxis.set_tick_params(direction='out', width=1, size=8)
    ax.yaxis.set_tick_params(direction='out', width=1, size=8)
    ax.xaxis.set_tick_params('minor', direction='out', width=1, size=4)
    ax.yaxis.set_tick_params('minor', direction='out', width=1, size=4)
    [sk.set_color('None') for k, sk in ax.spines.items() if k in 'top right'.split() ]
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

    ax.set_xlabel(xlabel, fontsize='x-large')
    ax.set_ylabel(ylabel, fontsize='x-large')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if fname is not None:
        plt.figtext(0.2, 0.9, r'{0:s}'.format(fname))
    return ax


def plot_models(x, y, logxnorm, logynorm, theta, ax=None,
                xlabel='X', ylabel='Y'):
    if ax is None:
        ax = plt.gca()

    logxmod = np.linspace(np.log10(x.min()), np.log10(x.max()), 300)

    # Trying to optimize plot transparencies
    plt_alpha = 0.4 / (5 * np.log10(len(theta)))

    ymods = []
    for tk in theta:
        alpha, beta, _ = tk
        logymod = (alpha * (logxmod - logxnorm) + beta + logynorm)
        ymods.append(10 ** logymod)
    ymods = np.array(ymods)

    ax.loglog(10 ** logxmod, ymods.T, color='#0088fa', alpha=plt_alpha,
              zorder=-100)

    ax.loglog(10 ** logxmod, np.median(ymods, 0), color='w', lw=2, zorder=-50)

    def switch_math(s):
        """ Make sure the text s is for math mode
        text need \\rm and math blocks nothing.
        """
        b = True
        txt = ''
        for k in s.split('$'):
            if (b & (len(k) > 0)):
                txt += r'{{\rm {0:s}}}'.format(k)
            elif (~b & (len(k) > 0)):
                txt += k
            b = ~b
        return txt

    def latex_float(f, precision=0.2, delimiter=r'\times'):
        """ Parse float to nice latex formatted string """
        float_str = ("{0:" + str(precision) + "g}").format(f)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            return (r"{0}" + delimiter + "10^{{{1}}}").format(base, int(exponent))
        else:
            return float_str

    ylabel.split('$')
    text = r'$\log_{10}\left(\frac{' + switch_math(ylabel) + r'}{'
    text += r'{ynorm:s}'.format(ynorm=latex_float(10 ** logynorm, 0.3))
    text += r'}\right) = \alpha\,\log_{10}\left(\frac{' + switch_math(xlabel) + r'}{'
    text += r'{xnorm:s}'.format(xnorm=latex_float(10 ** logxnorm, 0.3))
    text += r'}\right) + \beta$'

    plt.figtext(0.6, 0.15, text, fontsize='x-large')


def parse_config():
    """ parse the commandline for options and configuration file """
    import pyconfig
    import os

    opts = dict(
        default=(
            ('-n', '--nsamp',   dict(dest="number_of_samples", help="number of samples to represent per data point uncertainties", default=2000, type='int')),
            ('-c', '--config',   dict(dest="configfname", help="Configuration file to use for default values", default='None', type='str')),
        ),
        data=(
            ('-b', '--bootstrap', dict(dest="bootstrap", help="set to bootstrap the data", action='store_true', default=False)),
            ('--errorfloor', dict(dest='errorfloor', help="threshold under which errors are not reliable", default=1e-4, type='float')),
            ('--logxnorm',   dict(dest="log_xnorm", help="x-data normalization value", default=6, type='float')),
            ('--logynorm',   dict(dest="log_ynorm", help="y-data normalization value", default=0.5, type='float')),
            ('--xfloor',   dict(dest="xfloor", help="floor of x-value uncertainty (in %)", default=10, type='float')),
            ('--yfloor',   dict(dest="yfloor", help="floor of y-value uncertainty (in %)", default=10, type='float')),
        ),
        outputs=(
            ('-o', '--output',   dict(dest="output", help="export the samples into a file", default='None', type='str')),
            ('-f', '--savefig', dict(dest="plot", default='False',  help="Generate figures with the desired format (pdf, png...)", type='str')),
        ),
        plotting=(
            ('--xlabel',   dict(dest="xlabel", help="X-label of the plot (it can be in latex form)", default='X', type='str')),
            ('--ylabel',   dict(dest="ylabel", help="Y-label of the plot (it can be in latex form)", default='Y', type='str')),
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

    options.__dict__.pop('configfname')
    return args, options.__dict__


if __name__ == '__main__':
    args, opts = parse_config()
    output = opts.pop('output')

    theta = main_data(*args, **opts)

    if output not in ('None', 'none', None):
        print('Saving samples into {0:s}'.format(output))
        np.savetxt(output, theta, header='slope intercept dispersion')
