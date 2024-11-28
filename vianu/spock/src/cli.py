import argparse
import logging
import sys
from typing import List

from ..settings import DEFAULT_LOGGING_LEVEL, LOGGING_FMT, DEFAULT_DATA_DUMP
from . import scraping as scp
from . import chunking as cnk
from . import pipeline as ppl



MODULES = [scp, cnk, ppl]


def parse_args(args_):
    # Create global parser for logs
    global_parser = argparse.ArgumentParser(add_help=False)
    global_options = global_parser.add_argument_group('global')
    global_options.add_argument("--log-level", dest='log_level', default=DEFAULT_LOGGING_LEVEL)
    global_options.add_argument('--data-load', dest='data_load', help="File for initially loading the data")
    global_options.add_argument('--data-dump', dest='data_dump', default=DEFAULT_DATA_DUMP, help="File for saving pipeline progress")
    
    parser = argparse.ArgumentParser(description="SpoCK", parents=[global_parser])
    mod_parser = parser.add_subparsers(help='Module', dest="module", required=True)

    # Add single module parser
    for mod in MODULES:
        mod_parser.add_parser(mod.MODULE_NAME, parents=[global_parser, mod.cli_args()])
    
    # Add pipeline module parser
    parents = [global_parser, ppl.cli_args()] + [mod.cli_args() for mod in MODULES]
    mod_parser.add_parser(ppl.MODULE_NAME, parents=parents)

    args_ = parser.parse_args(args_)
    logging.basicConfig(level=args_.log_level.upper(), format=LOGGING_FMT)
    return args_


def main():
    args_= parse_args(sys.argv[1:])
    mod = [mod for mod in MODULES if mod.MODULE_NAME == args_.module][0]
    mod.apply(args_)


if __name__ == '__main__':
    main()
