from argparse import Namespace
import asyncio

from vianu.spock.__main__ import main
from vianu.spock.settings import N_SCP_TASKS_DEFAULT, N_NER_TASKS_DEFAULT, SCRAPING_SOURCES


_ARGS = {
    'term': 'dafalgan',
    'source': SCRAPING_SOURCES,
    'model': 'llama',
    'n_scp_tasks': N_SCP_TASKS_DEFAULT,
    'n_ner_tasks': N_NER_TASKS_DEFAULT,
    'log_level': 'DEBUG',
}


if __name__ == '__main__':
    asyncio.run(main(args_=Namespace(**_ARGS), save=False))
