""" trying a generic way to handle options from command line and config file """
from configparser import ConfigParser
from optparse import OptionParser, OptionGroup


def generate_conf_file_from_opts(*args):
    """ generate ascii corresponding to options that can be saved as a
    configuration file template

    Parameters
    ----------
    args: (name, options) tuples
        name of the section and options associated

    Returns
    -------
    txt: str
        configuration string
    """
    txt = ''

    # make a comment with help message and command line aliases
    # then a line with dest = value
    group_txt = '\n[{0:s}]\n'
    option_txt = '# {0} {1}\n{2} = {3}\n'

    for name, opt in args:
        txt += group_txt.format(name)
        for ko in opt:
            cnames = ko[:-1]
            cfg = ko[-1]
            if 'default' in cfg:
                txt += option_txt.format(cfg.get('help', None),
                                         cnames, cfg['dest'],
                                         cfg['default'])

    return txt


def make_parser_from_options(*args, **kwargs):
    """
    Parameters
    ----------
    args: (name, options) tuples
        name of the section and options associated

    kwargs: dict
        forwarded to :class:`OptionParser`

    Returns
    -------
    parser: OptionParser instance
        parser
    """
    parser = OptionParser(**kwargs)
    for name, opt in args:
        group = OptionGroup(parser, name)
        for ko in opt:
            group.add_option(*ko[:-1], **ko[-1])
        parser.add_option_group(group)
    return parser


def update_default_from_config_file(configfname, *args, **kwargs):
    """
    Parameters
    ----------
    configfname: str
        file containing the configuration

    args: (name, options) tuples
        name of the section and options associated

    kwargs: dict
        forwarded to :class:`OptionParser`

    Returns
    -------
    parser: OptionParser instance
        new parser
    """
    config = ConfigParser()
    config.read(configfname)

    parser = OptionParser(**kwargs)

    for name, opt in args:
        if name in config:
            kwargs = config[name]
            group = OptionGroup(parser, name)
            for ko in opt:
                cfg = ko[-1]
                if cfg['dest'] in kwargs:
                    cfg['default'] = kwargs[cfg['dest']]

                group.add_option(*ko[:-1], **ko[-1])
            parser.add_option_group(group)
    return parser


if __name__ == '__main__':

    # read config file first to get default values

    opts = dict(
        defaults=(
            ('-f', '--figure', dict(action="store_true", dest="make_figures", default=False, help="Generate figures at the end of the fit")),
            ('-o', '--output', dict(dest="output", default='tmp.hd5', type='str', help="file destination to save emcee sampler's outputs", metavar='FILE')),
            ('-i', '--input', dict(dest='fdata', default=None, type=str, help='input catalog file', metavar='FILE')),
            ('-q', '--quiet', dict(action="store_false", dest="verbose", default=True, help="don't print status messages to stdout")),
            ('--contours', dict(dest="contours", default='True', type='str', help="contour sequence on likelihood plots")),
            ('--vmax', dict(dest="vmax", default=5, type='float', help="max significance value on loglikelihood plot")),
            ('--config', dict(dest="configfname", help="Configuration file to use/save to for default values", default='None', type='str')),
        ),
        emcee=(
            ('-b', '--nburn', dict(dest="nburn", help="number of steps to burn in emcee", default=1500, type='int', metavar="int")),
            ('-n', '--nsample', dict(dest="nsample", help="number of steps to keep after burn in emcee", default=100, type='int', metavar="int")),
            ('-t', '--threads', dict(dest="threads", default=0, type='int', help="number of threads to run emcee", metavar='int')),
            ('-T', '--trim', dict(dest="trimchains", action="store_true", help="trim the chains half-way through burning phase")),
        ),
        model=(
            ('-N', '--ncfh', dict(dest="cfh_Nvals", help="number of bins in the piecewise constant CFH", default=10, type='int', metavar="int")),
            ('-c', '--completeness', dict(dest="complete_fn", default='ideal', type='str', help="Completeness model (ideal | empirical)", metavar='str')),
            ('-C', '--completeness_cut', dict(dest="complete_cut", default=None, type='float', help="Completeness threshold: anything below is discarded in the analysis")),
            ('--te', dict(dest="te", default=1e6, type='float', help="initial time at which starts of mass loss (arbitrary)")),
            ('--tmin', dict(dest="tage_min", default=1e6, type='float', help="lower age to consider in the model")),
            ('--tmax', dict(dest="tage_max", default=2e10, type='float', help="upper age to consider in the model")),
            ('--Mmin', dict(dest="Mmin", default=1e2, type='float', help="lower mass to consider in the model")),
            ('--Mmax', dict(dest="Mmax", default=1e7, type='float', help="upper age to consider in the model")),
            ('--dlogt', dict(dest="dlogt", default=0.05, type='float', help="binning in log(tage)")),
            ('--dlogm', dict(dest="dlogm", default=0.05, type='float', help="binning in log(M)")),
            ('--dlMi', dict(dest="dlMi", default=0.05, type='float', help="step size in the numerical integration over initial mass function")),
            ('-m', '--mask', dict(dest="mask", default=None, type='str', help="mask boundaries (tmin, tmax, Mmin, Mmax)")),
            ('-r', '--resample', dict(dest="data_nsamp", default=1, type='int', help="Number of samples from the data error to consider")),
        )
    )

    usage = "usage: %prog [options] JobID"
    parser = make_parser_from_options(*opts.items(), usage=usage)

    (options, args) = parser.parse_args()

    configfname = options.__dict__.pop('configfname')

    if configfname not in [None, 'None', 'none']:
        parser = update_default_from_config_file(configfname, *opts.items(),
                                                 usage=usage)

    (options, args) = parser.parse_args()

    print(generate_conf_file_from_opts(*opts.items()))
