from argparse import Namespace
import asyncio
from datetime import datetime
import logging
import os
import sys
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv

from vianu import LOG_FMT
from vianu.spock.settings import SCRAPING_SOURCES, LOG_LEVEL, MODEL_TEST_QUESTION
from vianu.spock.src.cli import parse_args
from vianu.spock.src.base import Setup, Document, SpoCK, FileHandler
from vianu.spock.src import scraping as scp
from vianu.spock.src import ner

logging.basicConfig(format=LOG_FMT, level=LOG_LEVEL)
logger = logging.getLogger(__name__)
load_dotenv()


def get_model_config() -> Dict[str, Dict[str, Any]]:
    """Get model configuration."""
    return {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "llama": {
            "base_url": os.getenv("OLLAMA_BASE_URL"),
        },
    }


async def _orchestrator(
    args_: Namespace,
    src_queue: asyncio.Queue,
    scp_queue: asyncio.Queue,
    ner_queue: asyncio.Queue,
    scp_tasks: List[asyncio.Task],
    ner_tasks: List[asyncio.Task],
) -> None:
    """Orchestrates the scraping and NER tasks.

    It waits for all scraping tasks to finish, then sends a sentinel to the scp_queue for each ner task (which will
    trigger the ner tasks to finish -> cf :func:`vianu.spock.src.ner.apply`).
    """
    logger.debug("setting up orchestrator task")

    # Insert sources into the source queue
    sources = args_.source
    for src in sources:
        await src_queue.put(src)

    # Insert sentinel for each scraping task
    for _ in range(len(scp_tasks)):
        await src_queue.put(None)

    # Wait for all scraper tasks to finish and stop them
    await src_queue.join()
    try:
        await asyncio.gather(*scp_tasks)
    except asyncio.CancelledError:
        logger.warning("scraping task(s) have previously been canceled")
    except Exception as e:
        logger.error(f"scraping task(s) failed with error: {e}")
        raise e
    for st in scp_tasks:
        st.cancel()

    # Insert sentinel for each NER
    for _ in range(len(ner_tasks)):
        await scp_queue.put(None)

    # Wait for NER tasks to process all items and finish
    await scp_queue.join()
    try:
        await asyncio.gather(*ner_tasks)
    except asyncio.CancelledError:
        logger.warning("ner task(s) have previously been canceled")
    except Exception as e:
        logger.error(f"ner task(s) failed with error: {e}")
        raise e
    for nt in ner_tasks:
        nt.cancel()

    # Insert sentinel into ner_queue to indicate end of processing
    await ner_queue.put(None)


def setup_asyncio_framework(
    args_: Namespace, model_config: Dict[str, Any]
) -> Tuple[asyncio.Queue, List[asyncio.Task], List[asyncio.Task], asyncio.Task]:
    """Set up the asyncio framework for the SpoCK application."""
    # Set up arguments
    if args_.source is None:
        args_.source = SCRAPING_SOURCES

    # Set up queues
    src_queue = asyncio.Queue()
    scp_queue = asyncio.Queue()
    ner_queue = asyncio.Queue()

    # Start tasks
    scp_tasks = scp.create_tasks(args_=args_, queue_in=src_queue, queue_out=scp_queue)
    ner_tasks = ner.create_tasks(
        args_=args_, queue_in=scp_queue, queue_out=ner_queue, model_config=model_config
    )
    orc_task = asyncio.create_task(
        _orchestrator(
            args_=args_,
            src_queue=src_queue,
            scp_queue=scp_queue,
            ner_queue=ner_queue,
            scp_tasks=scp_tasks,
            ner_tasks=ner_tasks,
        )
    )
    return ner_queue, scp_tasks, ner_tasks, orc_task


async def _collector(ner_queue: asyncio.Queue) -> List[Document]:
    """Collect results from the NER queue."""
    data = []
    while True:
        item = await ner_queue.get()

        # Check stopping condition
        if item is None:
            ner_queue.task_done()
            break

        # Append document to data
        data.append(item.doc)
        ner_queue.task_done()

    return data


async def main(args_: Namespace | None = None, save: bool = True) -> None:
    """Main function for the SpoCK pipeline."""
    started_at = datetime.now()
    if args_ is None:
        args_ = parse_args(sys.argv[1:])

    logging.basicConfig(level=args_.log_level.upper(), format=LOG_FMT)
    logger.info(f"starting SpoCK (args_={args_})")

    # Test availability of NER model
    model = args_.model
    model_config = get_model_config()
    try:
        _ner = ner.NERFactory.create(model=model, config=model_config)
        test_task = asyncio.create_task(_ner.test_model_endpoint())
        test_answer = await test_task
        logger.debug(
            f"test model endpoint of model='{model}': '{MODEL_TEST_QUESTION}' was answered with '{test_answer}'"
        )
    except Exception as e:
        logger.error(f"could not reach model endpoint: {e}")
        raise e

    # Set up async structure (scraping queue/tasks, NER queue/tasks, orchestrator task)
    ner_queue, _, _, _ = setup_asyncio_framework(args_=args_, model_config=model_config)

    # Set up collector task and wait for it to finish
    # NOTE: if collector task is finished, the orchestrator is also finished (because of the sentinel in `ner_queue`)
    # and therefore so are the scraping and NER tasks
    col_task = asyncio.create_task(_collector(ner_queue))
    data = await col_task
    await ner_queue.join()

    # Save data
    if save:
        file_name = args_.file_name
        file_path = args_.file_path
        spock = SpoCK(
            id_=str(args_),
            status="completed",
            started_at=started_at,
            finished_at=datetime.now(),
            setup=Setup.from_namespace(args_),
            data=data,
        )
        if file_name is not None and file_path is not None:
            FileHandler(file_path=file_path).write(file_name=file_name, spock=spock)
    logger.info("finished SpoCK")


if __name__ == "__main__":
    asyncio.run(main())
