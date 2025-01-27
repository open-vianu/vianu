"""CLI for SpoCK"""

import argparse
from typing import Sequence

from vianu.spock.settings import LOG_LEVEL, FILE_NAME, FILE_PATH, MAX_DOCS_SRC
from vianu.spock.settings import SCRAPING_SOURCES, N_SCP_TASKS
from vianu.spock.settings import N_NER_TASKS, LARGE_LANGUAGE_MODELS


def parse_args(args_: Sequence) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpoCK", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add generic options
    gen_gp = parser.add_argument_group("generic")
    gen_gp.add_argument(
        "--log-level", metavar="", type=str, default=LOG_LEVEL, help="log level"
    )
    gen_gp.add_argument(
        "--file-path",
        metavar="",
        type=str,
        default=FILE_PATH,
        help="path for storing results",
    )
    gen_gp.add_argument(
        "--file-name",
        metavar="",
        type=str,
        default=FILE_NAME,
        help="filename for storing results",
    )
    gen_gp.add_argument(
        "--max-docs-src",
        metavar="",
        type=int,
        default=MAX_DOCS_SRC,
        help="maximum number of documents per source",
    )

    # Add scraping group
    scp_gp = parser.add_argument_group("scraping")
    scp_gp.add_argument(
        "--source",
        "-s",
        type=str,
        action="append",
        choices=SCRAPING_SOURCES,
        help="data sources for scraping",
    )
    scp_gp.add_argument("--term", "-t", metavar="", type=str, help="search term")
    scp_gp.add_argument(
        "--n-scp-tasks",
        metavar="",
        type=int,
        default=N_SCP_TASKS,
        help="number of async scraping tasks",
    )

    # Add NER group
    ner_gp = parser.add_argument_group("ner")
    ner_gp.add_argument(
        "--model",
        "-m",
        type=str,
        choices=LARGE_LANGUAGE_MODELS,
        default="openai",
        help="NER model",
    )
    ner_gp.add_argument(
        "--n-ner-tasks",
        metavar="",
        type=int,
        default=N_NER_TASKS,
        help="number of async ner tasks",
    )

    return parser.parse_args(args_)
