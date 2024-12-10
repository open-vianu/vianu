from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from typing import Iterable, List

from .data_model import FileHandler
from . import scraping as scp
from . import ner

logger = logging.getLogger(__name__)

MODULE_NAME = 'pipeline'
PIPELINE_STEPS = [scp, ner] 

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
    return parser

def apply(args_: Namespace, save_data: bool = True) -> None:
    logger.info('run pipeline')

    data = scp.apply(args_=args_, save_data=False)
    ner.apply(args_=args_, data=data, save_data=False)

    if save_data:
        FileHandler(args_.data_dump).write(data)
