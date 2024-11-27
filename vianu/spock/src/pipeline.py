from argparse import ArgumentParser, ArgumentTypeError
from typing import List

from . import scraping as scp
from . import chunking as cnk

MODULE_NAME = 'pipeline'
PIPELINE_STEPS = {
    'all': [scp, cnk],
    'scp': scp,
    'cnk': cnk, 
}

def _pipeline_steps(steps):
    def _parse(arg):
        values = arg.split(',')
        for val in values:
            if val not in steps:
                raise ArgumentTypeError(f"Invalid choice: {val} (choose from {', '.join(steps)})")
        return values
    return _parse

def cli_args():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    group.add_argument('--steps', dest='steps', type=_pipeline_steps(PIPELINE_STEPS.keys()), required=True)
    return parser

def apply(args_):
    if 'all' in args_.steps:
        steps = PIPELINE_STEPS['all']
    else:
        steps = [PIPELINE_STEPS[s] for s in args_.steps]
    print(steps)
    

