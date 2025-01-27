"""CLI for LASA"""

import argparse
from typing import Sequence

from vianu.lasa.settings import LOG_LEVEL, SOURCES


def parse_args(args_: Sequence) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LASA: Loocks Alike Sounds Alike Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-level", metavar="", type=str, default=LOG_LEVEL, help="log level"
    )
    parser.add_argument(
        "--search", "-s", metavar="", type=str, required=True, help="search term"
    )
    parser.add_argument(
        "--source",
        "-r",
        metavar="",
        type=str,
        action="append",
        choices=SOURCES,
        help="source of authorized medicines registry",
    )

    return parser.parse_args(args_)
