from argparse import Namespace
import logging
from multiprocessing import Pool
import sys

from typing import Iterable, List

from vianu import LOG_FMT
from vianu.lasa.settings import LOG_LEVEL, PROCESSES, SOURCES
from vianu.lasa.src.base import LASA, Match
from vianu.lasa.src.units import AuthorizationFactory
from vianu.lasa.src.cli import parse_args

logging.basicConfig(level=LOG_LEVEL.upper(), format=LOG_FMT)
logger = logging.getLogger(__name__)


def process(
    lasa: LASA, names: Iterable[str], processes: int = PROCESSES
) -> List[Match]:
    """Perform the LASA matches in parallel with a pool of processes"""
    logger.debug(
        f"matching ref={lasa._ref} against #names={len(names)} with #processes={PROCESSES}"
    )
    with Pool(processes=processes) as pool:
        matches = pool.map(func=lasa.apply, iterable=names)
    logger.debug(f"matching completed for ref={lasa._ref}")
    return matches


def main(args_: Namespace | dict | None = None) -> List[Match]:
    """Main entry point for LASA"""
    if args_ is None:
        args_ = parse_args(sys.argv[1:])
    elif isinstance(args_, dict):
        args_ = Namespace(**args_)
    if args_.source is None:
        args_.source = SOURCES

    logger.info(f"starting LASA with args_={args_}")
    lasa = LASA(ref=args_.search)

    matches = []
    for src in args_.source:
        authorization = AuthorizationFactory.create(src)
        names = authorization.get_products_as_df()["name"].to_list()
        logger.debug(f"using source={src} it produced #names={len(names)}")

        src_mtch = process(lasa=lasa, names=names)
        for mt in src_mtch:
            mt.source = src
        matches.extend(src_mtch)
    logger.info("LASA completed")
    return matches
