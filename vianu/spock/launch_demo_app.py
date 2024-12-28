import asyncio
from copy import deepcopy
from datetime import datetime
import logging
from pathlib import Path
from typing import Tuple, List, Any

import gradio as gr

from vianu.spock.settings import LOGGING_LEVEL, LOGGING_FMT, UPDATE_INTERVAL, MAX_JOBS
from vianu.spock.src.data_model import Job, SpoCK, QueueItem
from vianu.spock.__main__ import setup_asyncio_framework
from vianu.spock.src.ui import get_job_card_html, get_details_html


logging.basicConfig(level=LOGGING_LEVEL.upper(), format=LOGGING_FMT)
logger = logging.getLogger(__name__)

HEAD_FILE = Path(__file__).parent / "assets/head/scripts.html"
CSS_FILE = Path(__file__).parent / "assets/css/styles.css"
SPOCK_KWARGS = {
    "source": ["pubmed"], 
    "model": "llama",
    "n_scp_tasks": 1,
    "n_ner_tasks": 1,
    "log_level": LOGGING_LEVEL,
}

# Global async variables
ner_queue = None
scp_tasks = None
ner_tasks = None
orc_task = None
col_task = None

# Global SpoCK variables
job = None
spocks = []
running_spock = None


async def _toggle_button(is_running: bool) -> Tuple[bool, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Toggle the state of the pipleline between running <-> not running. As a result the corresponding buttons are 
    shown/hidden.
    """
    logger.debug(f'toggle button (is_running={is_running}->{not is_running})')
    is_running = not is_running
    if is_running:
        # Show the stop button and hide the start/cancel button
        return is_running, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    else:
        # Show the start button and hide the stop/cancel button
        return is_running, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


async def _show_cancel_button() -> Tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Shows the cancel button and hides the start and stop button."""
    logger.debug('show cancel button')
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


async def _setup_spock(term: str) -> None:
    """Setup the SpoCK (Job and Data) object and assign it to the global variable running_spock."""
    global running_spock, spocks, job

    # Check if the maximum number of jobs is reached (gradio <= 5.0.0 does not support dynamic number of gr.Blocks)
    if len(spocks) >= MAX_JOBS:
        msg = f"max number of jobs reached: {MAX_JOBS}"
        gr.Warning(msg)
        logger.warning(msg)
        return None
    
    # Setup the running_spock and append it to the list of spocks
    msg = f'started SpoCK for "{term}"'
    gr.Info(msg)
    logger.info(msg)

    args_ = deepcopy(SPOCK_KWARGS)
    args_["term"] = term
    job = Job(id_=f'{args_["term"]} {args_["source"]} {args_["model"]}', **args_)
    running_spock = SpoCK(status='running', job=job, started_at=job.submission, data=[])
    spocks.insert(0, running_spock)


async def _collector() -> None:
    """Append the processed document from the NER queue to `running_spock.data`."""
    global ner_queue, running_spock

    logger.debug(f'starting collector (term={running_spock.job.term})')
    if len(spocks) >= MAX_JOBS:
        return
    while True:
        item = await ner_queue.get()    # type: QueueItem
        # Check stopping condition (added by the `orchestrator` in `vianu.spock.__main__`)
        if item is None:
            ner_queue.task_done()
            break
        running_spock.data.append(item.doc)
        ner_queue.task_done()


async def _setup_asyncio_framework() -> None:
    """"Start the SpoCK processes by setting up the asyncio framework and starting the asyncio tasks.
    
    Main components of asyncio framework are:
    - ner_queue: queue for collecting results from named entity recognition tasks
    - scp_tasks: scraping tasks
    - ner_tasks: named entity recognition tasks
    - orc_task: orchestrating the scraping and named entity recognition tasks
    - col_task: collecting the results from the named entity recognition tasks
    """
    global ner_queue, scp_tasks, ner_tasks, orc_task, col_task

    if len(spocks) >= MAX_JOBS:
        return None
    logger.info("setting up asyncio framework")
    ner_queue, scp_tasks, ner_tasks, orc_task = setup_asyncio_framework(args_=job)
    col_task = asyncio.create_task(_collector())


async def _feed_cards_to_ui() -> List[dict[str, Any]]:
    """From the current and previous SpoCKs, create and feed the job cards to the UI."""
    global spocks

    logger.debug(f'feeding cards to UI (len(spocks)={len(spocks)})')
    # Create the job cards for the existing spocks
    cds = []
    for i, spk in enumerate(spocks):
        html = get_job_card_html(i, spk.job, spk.data)
        cds.append(gr.update(value=html, visible=True))

    # Extdend with empty cards (as dynamic number of gr.Blocks is not supported in gradio <= 5.0.0)
    cds.extend([gr.update(visible=False) for i in range(MAX_JOBS - len(spocks))])
    return cds


async def _conclusion():
    global ner_queue, orc_task, col_task, running_spock

    # Wait collector task to finish and join ner_queue
    try:
        await col_task
    except asyncio.CancelledError:
        logger.warning('collector task canceled')
        return None   # This stops the _conclusion step in the case the _canceling step was triggered
    except Exception as e:
        logger.error(f'collector task failed with error: {e}')
        raise e
    await ner_queue.join()
    
    # Update the running_spock with the final data
    running_spock.status = 'completed'
    running_spock.finished_at = datetime.now()

    # Log the conclusion and update/empty the running_spock
    gr.Info(f'job "{running_spock.job.term}" finished')
    logger.info(f'job "{running_spock.job.term}" finished in {running_spock.runtime()}')


async def _canceling():
    """Cancel all running :class:`asyncio.Task`."""
    global scp_tasks, ner_tasks, orc_task, col_task, running_spock

    gr.Warning(f'canceled SpoCK for "{running_spock.job.term}"')
    running_spock.status = 'stopped'
    running_spock.finished_at = datetime.now()

    # Cancel scraping tasks
    logger.warning("canceling scraping tasks")
    for task in scp_tasks:
        task.cancel()
    await asyncio.gather(*scp_tasks, return_exceptions=True)

    # Cancel named entity recognition tasks
    logger.warning("canceling named entity recognition tasks")
    for task in ner_tasks:
        task.cancel()
    await asyncio.gather(*ner_tasks, return_exceptions=True)

    # Cancel orchestrator task
    logger.warning("canceling orchestrator task")
    orc_task.cancel()
    await asyncio.gather(orc_task, return_exceptions=True)    # we use return_exceptions=True to avoid raising exceptions due to the subtasks being canceled`

    # Cancel collector task
    logger.warning("canceling collector task")
    col_task.cancel()
    await asyncio.gather(col_task, return_exceptions=True)    # see remark above


def _change_card_number(icrd: int):
    logger.debug(f'card clicked={icrd}')
    return icrd


def _feed_details_to_ui(icrd: int):
    """Collect the (previously created) html texts for the documents of the selected job and feed them to the UI."""
    global spocks
    if len(spocks) == 0:
        return get_details_html([])
    return get_details_html(spocks[icrd].data)


# Design of the UI
with gr.Blocks(head_paths=HEAD_FILE, css_paths=CSS_FILE, theme=gr.themes.Soft()) as demo:
    is_running = gr.State(value=False)
    card_number = gr.State(value=0)

    # Logo and title
    with gr.Row(elem_id="logo-title-row"):
        with gr.Column(scale=1):
            gr.Image(
                value="vianu/spock/assets/images/spock_logo_circular.png",
                show_label=False,
                elem_id="logo-image",
            )
        with gr.Column(scale=5):
            gr.Markdown("<div id='title-text'>SpoCK: Spotting Clinical Knowledge</div>")
    
    # Search text field and start/stop/cancel buttons
    with gr.Row(elem_classes="search-container"):        
        with gr.Column(scale=3):
            search_term = gr.Textbox(label="Search", show_label=False, placeholder="Enter your search term", )
        with gr.Column(scale=1, elem_classes='pipeline-button'):
            start_button = gr.HTML('<div class="button-not-running">Start</div>', visible=True)
            stop_button = gr.HTML('<div class="button-running">Stop</div>', visible=False)
            cancel_button = gr.HTML('<div class="canceling">canceling...</div>', visible=False)

    # Job summary cards
    with gr.Row(elem_classes="jobs-container"):
        cards = [gr.HTML('', elem_id=f'job-{i}', visible=False) for i in range(MAX_JOBS)]

    # Details of the selected job
    with gr.Row():
        details = gr.HTML('<div class="details-container"></div>')
    
    # Timer for automatic update of the available documents for the running job
    feed_details_timer = gr.Timer(value=UPDATE_INTERVAL, active=False, render=True)
    feed_details_timer.tick(fn=_feed_details_to_ui, inputs=card_number, outputs=details)
    
    # Starting the pipeline with the search term
    start_button.click(
        fn=_toggle_button, inputs=is_running, outputs=[is_running, start_button, stop_button, cancel_button]
    ).then(
        fn=lambda: gr.update(active=True), outputs=feed_details_timer
    ).then(
        fn=_setup_spock, inputs=search_term
    ).then(
        fn=_setup_asyncio_framework
    ).then(
        fn=lambda: None, outputs=search_term    # Empty the search term in the UI
    ).then(
        fn=_feed_cards_to_ui, outputs=cards
    ).then(
        fn=_conclusion
    ).then(
        fn=lambda: gr.update(active=False), outputs=feed_details_timer
    ).then(
        fn=_toggle_button, inputs=[is_running], outputs=[is_running, start_button, stop_button, cancel_button]
    )

    # Stopping the pipeline
    stop_button.click(
        fn=_show_cancel_button, outputs=[start_button, stop_button, cancel_button]
    ).then(
        fn=_canceling
    )    # NOTE thath "toggling the button" and "setting timer active=False" is not needed: '_canceling' is terminating the still running '_conclusion' step which is followed by these two steps
    
    # Callback for the job cars to show the details
    for icrd, crd in enumerate(cards):
        crd.click(
            fn=_change_card_number, inputs=gr.Number(value=icrd, visible=False), outputs=card_number
        ).then(
            fn=_feed_details_to_ui, inputs=card_number, outputs=details
        )


if __name__ == "__main__":
    demo.launch(debug=True)
