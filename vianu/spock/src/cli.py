import argparse
import logging
import sys  

from ..settings import DEFAULT_LOGGING_LEVEL, LOGGING_FMT
from . import scraping as scp
from . import chunking as cnk


MODULES = [scp, cnk]


def parse_args(args_):    # Create global parser for logs

    global_parser = argparse.ArgumentParser(add_help=False)
    global_options = global_parser.add_argument_group('Global options')
    global_options.add_argument("--log-level", dest='log_level', default=DEFAULT_LOGGING_LEVEL)
    
    parser = argparse.ArgumentParser(description="SpoCK", parents=[global_parser])
    mod_parser = parser.add_subparsers(help='Module', dest="module", required=True)

    # Add parser
    for mod in MODULES:
        mod.add_parser(subparser=mod_parser, parents=[global_parser])
    
    args_ = parser.parse_args(args_)
    logging.basicConfig(level=args_.log_level.upper(), format=LOGGING_FMT)
    return args_


def main():
    args_= parse_args(sys.argv[1:])
    mod = [mod for mod in MODULES if mod.MODULE_NAME == args_.module][0]
    mod.apply(args_)


if __name__ == '__main__':
    main()