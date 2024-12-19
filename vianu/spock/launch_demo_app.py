import asyncio
from copy import deepcopy
import logging
from pathlib import Path

import gradio as gr

from vianu.spock.settings import LOGGING_FMT, LOGGING_LEVEL
from vianu.spock.src.data_model import Job, SpoCK
from vianu.spock.__main__ import setup_asyncio_framework
from vianu.spock.src.frontend import format_job_card, get_details_of_data


logging.basicConfig(level=LOGGING_LEVEL.upper(), format=LOGGING_FMT)
logger = logging.getLogger(__name__)

HEAD_FILE = Path(__file__).parent / "assets/head/scripts.html"
CSS_FILE = Path(__file__).parent / "assets/css/styles.css"
SPOCK_KWARGS = {
    "source": ["pubmed"], 
    "model": "llama", 
    "n_ner_tasks": 1,
    "log_level": LOGGING_LEVEL,
}
MAX_JOBS = 5

# Global variables
ner_queue = None
orc_task = None
job = None
cards = []
spocks = []
running = None


async def setup_spock(term: str):
    global running, spocks, job
    if len(spocks) >= MAX_JOBS:
        logger.error(f"Max number of jobs reached: {MAX_JOBS}")
        return None
    logger.info(f"Setting up SpoCK for term: {term}")
    args_ = deepcopy(SPOCK_KWARGS)
    args_["term"] = term
    job = Job(id_=f'{args_["term"]} {args_["source"]} {args_["model"]}', **args_)
    running = SpoCK(job=job, started_at=job.submission, data=[])
    print(f'running exists len data={len(running.data)}')
    spocks.append(running)


async def start_processes():
    global ner_queue, orc_task
    if len(spocks) >= MAX_JOBS:
        return None
    logger.info("Starting SpoCK processes")
    _, ner_queue, _, _, orc_task = setup_asyncio_framework(args_=job)


async def add_cards():
    global spocks
    cards = []
    for i, spk in enumerate(spocks):
        html = format_job_card(i, spk.job, spk.data)
        cards.append(gr.HTML(html, elem_id=f"job-{i}", visible=True))
    cards.extend([gr.HTML('', elem_id=f'job-{len(spocks) + i}', visible=False) for i in range(MAX_JOBS - len(spocks))])
    return cards


async def add_data_to_spock():
    global ner_queue, running
    if len(spocks) >= MAX_JOBS:
        return
    while True:
        item = await ner_queue.get()
        if item is None:
            break
        running.data.append(item.doc)


async def get_details(job_id: str):
    global spocks
    i = int(job_id.split("-")[-1])
    while True:
        await asyncio.sleep(2)
        data = get_details_of_data(spocks[i].data)
        yield data



# Layout resembling the image
with gr.Blocks(head_paths=HEAD_FILE, css_paths=CSS_FILE, theme=gr.themes.Soft()) as demo:
    # scraping = gr.State(value=False)
    # nering = gr.State(value=False)

    with gr.Row(elem_id="logo-title-row"):
        with gr.Column(scale=1):
            gr.Image(
                value="vianu/spock/assets/images/spock_logo_circular.png",
                show_label=False,
                elem_id="logo-image",
            )
        with gr.Column(scale=5):
            gr.Markdown("<div id='title-text'>SpoCK: Spotting Clinical Knowledge</div>")
    
    with gr.Row():
        search_input = gr.Textbox(label="Search", placeholder="Enter your search here...", )

    with gr.Row(elem_classes="jobs-container"):
        cards = [gr.HTML('', elem_id=f'job-{i}', visible=False) for i in range(MAX_JOBS)]

    with gr.Row():
        details = gr.HTML('<div id="details" class="details-container"></div>')

    search_input.submit(
        fn=setup_spock, inputs=search_input
    ).then(
        fn=start_processes
    ).then(
        fn=add_cards, outputs=cards
    ).then(
        fn=add_data_to_spock
    )
    
    for i, crd in enumerate(cards):
        crd.click(fn=get_details, inputs=gr.Textbox(value=crd.elem_id, visible=False), outputs=details)


if __name__ == "__main__":
    demo.launch(debug=True)
