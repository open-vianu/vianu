import argparse
import logging
import sys

from ..settings import LOGGING_LEVEL, LOGGING_FMT, DATA_FILW
from . import scraping as scp
from . import ner
from . import pipeline as ppl



MODULES = [scp, ner, ppl]
_MOD_NON_PPL = [mod for mod in MODULES if mod.MODULE_NAME != ppl.MODULE_NAME]

def parse_args(args_: argparse.Namespace) -> argparse.Namespace:
    # Create global parser for logs
    global_parser = argparse.ArgumentParser(add_help=False)
    global_options = global_parser.add_argument_group('global')
    global_options.add_argument("--log-level", dest='log_level', default=LOGGING_LEVEL)
    global_options.add_argument('--data-load', dest='data_load', help="File for initially loading the data")
    global_options.add_argument('--data-dump', dest='data_dump', default=DATA_FILW, help="File for saving pipeline progress")
    
    parser = argparse.ArgumentParser(description="SpoCK", parents=[global_parser])
    mod_parser = parser.add_subparsers(help='Module', dest="module", required=True)

    # Add single module parser
    for mod in _MOD_NON_PPL:
        mod_parser.add_parser(mod.MODULE_NAME, parents=[global_parser, mod.cli_args()])
    
    # Add pipeline module parser
    parents = [global_parser, ppl.cli_args()] + [mod.cli_args() for mod in _MOD_NON_PPL]
    mod_parser.add_parser(ppl.MODULE_NAME, parents=parents)

    return parser.parse_args(args_)


def main():
    args_= parse_args(sys.argv[1:])
    logging.basicConfig(level=args_.log_level.upper(), format=LOGGING_FMT)
    
    mod = [mod for mod in MODULES if mod.MODULE_NAME == args_.module][0]
    mod.apply(args_)


if __name__ == '__main__':
    main()
