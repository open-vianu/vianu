import logging

from .src.cli import main
from .settings import LOGGING_LEVEL, LOGGING_FMT

logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FMT)

main()