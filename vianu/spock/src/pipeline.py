from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from pathlib import Path
from typing import Iterable, List

from .data_model import FileHandler
from . import scraping as scp
from . import chunking as cnk
from . import ner

MODULE_NAME = 'pipeline'
PIPELINE_STEPS = {
    'all': [scp, cnk, ner],
    'scp': scp,
    'cnk': cnk,
    'ner': ner,
}

def _pipeline_steps(steps: Iterable[str]):
    def _parse(arg: str) -> List[str]:
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

def apply(args_: Namespace):
    logging.info(f'run pipeline seps {args_.steps}')
    if 'all' in args_.steps:
        steps = PIPELINE_STEPS['all']
    else:
        steps = [PIPELINE_STEPS[s] for s in args_.steps]
    
    if (data_load := args_.data_load) is not None:
        data_file = Path(data_load)
        data = FileHandler(data_file=data_file).read()
    else:
        data = []
    
    for stp in steps:
        stp.apply(args_, data)
    
    FileHandler(args_.data_dump).write(data)
