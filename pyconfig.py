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
