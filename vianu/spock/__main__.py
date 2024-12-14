import asyncio
import logging
import sys
from typing import List

from .src.cli import parse_args
from .src.data_model import FileHandler
from .src import scraping as scp
from .src import ner
from .settings import LOGGING_FMT


async def orchestrator(
        scp_tasks: List[asyncio.Task],
        ner_tasks: List[asyncio.Task],
        scp_queue: asyncio.Queue,
        ner_queue: asyncio.Queue, 
        ) -> None:
    
    # Wait for all scraper tasks to finish and stop them
    await asyncio.gather(*scp_tasks)
    for st in scp_tasks:
        st.cancel()

    # Insert sentinel for each NER
    for _ in range(len(ner_tasks)):
        await scp_queue.put(None)
    
    # Wait for NER tasks to process all items and finish
    await scp_queue.join()
    await asyncio.gather(*ner_tasks)
    for nt in ner_tasks:
        nt.cancel()

    # Insert sentinel into ner_queue to indicate no more results
    await ner_queue.put(None)



async def main():
    args_= parse_args(sys.argv[1:])
    logging.basicConfig(level=args_.log_level.upper(), format=LOGGING_FMT)
    logging.info(f'Starting SpoCK (args_={args_})')    
    
    # Set up queues
    scp_queue = asyncio.Queue()
    ner_queue = asyncio.Queue()

    # Start tasks
    n_tasks = args_.n_tasks
    scp_tasks = scp.create_tasks(args_=args_, queue=scp_queue)
    ner_tasks = ner.create_tasks(args_=args_, queue_in=scp_queue, queue_out=ner_queue, n_tasks=n_tasks)
    orc_task = asyncio.create_task(orchestrator(scp_tasks, ner_tasks, scp_queue, ner_queue))

    # Read results from NER queue
    data = []
    while True:
        doc = await ner_queue.get()

        # Check stopping condition
        if doc is None:
            ner_queue.task_done()
            break

        # Append document to data
        data.append(doc)
        ner_queue.task_done()
    
    # Wait for orchestrator to finish and queue to be empty
    await orc_task
    await ner_queue.join()

    # Save data
    FileHandler(args_.data_file).write(data)

    logging.info('Finished SpoCK')


if __name__ == '__main__':
    asyncio.run(main())
