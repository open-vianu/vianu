"""CLI for SpoCK
"""

import argparse

from vianu.spock.settings import LOGGING_LEVEL, DATA_FILE, DATA_PATH, SCRAPING_SOURCES, N_TASKS_DEFAULT, NER_MODELS


def parse_args(args_: argparse.Namespace) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpoCK", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add generic options
    gen_gp = parser.add_argument_group('generic')
    gen_gp.add_argument("--log-level", metavar='', type=str, default=LOGGING_LEVEL, help='log level')
    gen_gp.add_argument("--data-path", metavar='', type=str, default=DATA_PATH, help='path for storing results')
    gen_gp.add_argument("--data-file", metavar='', type=str, default=DATA_FILE, help='filename for storing results')

    # Add scraping group
    scp_gp = parser.add_argument_group('scraping')
    scp_gp.add_argument('--source', '-s', type=str, action='append', choices=SCRAPING_SOURCES, help='data sources for scraping')
    scp_gp.add_argument('--term', '-t', metavar='', type=str, help='search term')

    # Add NER group
    ner_gp = parser.add_argument_group('ner')
    ner_gp.add_argument('--model', '-m', type=str, choices=NER_MODELS, default='llama', help='NER model')
    ner_gp.add_argument('--n-ner-tasks', metavar='', type=int, default=N_TASKS_DEFAULT, help='number of async ner tasks')

    return parser.parse_args(args_)
