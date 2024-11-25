from argparse import ArgumentParser
import logging

from .data_model import DOCUMENT_SOURCES


MODULE_NAME = 'scraping'

def add_parser(parent: ArgumentParser):
    subparser = parent.add_parser(MODULE_NAME)
    subparser.add_argument('--source', choices=DOCUMENT_SOURCES)


def apply():
    logging.info(f'running {MODULE_NAME.upper()} module')
    pass
