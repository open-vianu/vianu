from copy import deepcopy
import logging
from pathlib import Path

import gradio as gr

from vianu.spock.settings import LOGGING_FMT, LOGGING_LEVEL
from vianu.spock.src.data_model import Job, SpoCK
from vianu.spock.__main__ import setup_framework
from vianu.spock.src.frontend import format_job_card, _get_details_data, CARD_CONTAINER_TEMPLATE


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
ner_queue = None
orc_task = None
job = None
spocks = []


async def setup_spock(term: str):
    global spocks, job
    logger.info(f"Setting up SpoCK for term: {term}")
    args_ = deepcopy(SPOCK_KWARGS)
    args_["term"] = term
    job = Job(id_=f'{args_["term"]} {args_["source"]} {args_["model"]}', **args_)
    spock = SpoCK(job=job, started_at=job.submission, data=[])
    spocks.append(spock)


async def start_processes():
    global ner_queue, orc_task
    logger.info("Starting SpoCK processes")
    _, ner_queue, _, _, orc_task = setup_framework(args_=job)


async def get_cards():
    cards = [format_job_card(i, spk.job, spk.data) for i, spk in enumerate(spocks)]
    return CARD_CONTAINER_TEMPLATE.format(cards="\n".join(cards))


async def get_details():
    global ner_queue, spocks
    spock = spocks[-1]
    while True:
        item = await ner_queue.get()
        if item is None:
            break
        spock.data.append(item.doc)
        yield _get_details_data(spock.data)


# Layout resembling the image
with gr.Blocks(head_paths=HEAD_FILE, css_paths=CSS_FILE, theme=gr.themes.Soft()) as demo:
    scraping = gr.State(value=False)
    nering = gr.State(value=False)

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
        search_input = gr.Textbox(label="Search", placeholder="Enter your search here...")

    with gr.Row():
        cards = gr.HTML('<div id="cards" class="cards-container"></div>')

    with gr.Row():
        details = gr.HTML('<div id="details" class="details-container"></div>')
        # details = gr.TextArea(label="Details", placeholder="Details will appear here...")

    search_input.submit(
        fn=setup_spock, inputs=search_input
    ).then(
        fn=start_processes
    ).then(
        fn=get_cards, inputs=None, outputs=cards
    ).then(
        fn=get_details, outputs=details
    )


if __name__ == "__main__":
    demo.launch(debug=True)
