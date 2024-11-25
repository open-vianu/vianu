import argparse
import sys  

from . import scraping as scp
from . import pipeline as ppl


MODULES = [scp]


def parse_args(args_):
    parent = argparse.ArgumentParser(description="SpoCK")
    mod_parser = parent.add_subparsers(help='Module', dest="module", required=True)

    # Add parser
    for mod in MODULES:
        mod.add_parser(parent=mod_parser)
    
    return parent.parse_args(args_)


def main():
    args_= parse_args(sys.argv[1:])
    mod = [mod for mod in MODULES if mod.MODULE_NAME == args_.module][0]
    mod.apply(args_)


if __name__ == '__main__':
    main()