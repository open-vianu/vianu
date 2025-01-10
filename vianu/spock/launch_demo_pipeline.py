from argparse import Namespace
import asyncio

from vianu.spock.__main__ import main
from vianu.spock.settings import SCRAPING_SOURCES, MAX_DOCS_SRC
from vianu.spock.settings import N_SCP_TASKS_DEFAULT, N_NER_TASKS_DEFAULT


_ARGS = {
    'term': 'ibuprofen',
    'max_docs_src': MAX_DOCS_SRC,
    'source': SCRAPING_SOURCES,
    'model': 'llama',
    'n_scp_tasks': N_SCP_TASKS_DEFAULT,
    'n_ner_tasks': N_NER_TASKS_DEFAULT,
    'log_level': 'DEBUG',
}


if __name__ == '__main__':
    asyncio.run(main(args_=Namespace(**_ARGS), save=False))
