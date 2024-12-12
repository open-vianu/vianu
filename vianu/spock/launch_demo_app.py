from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import List

import gradio as gr

from vianu.spock.src import scraping as scp
from vianu.spock.src import ner
from vianu.spock.src.data_model import Document, Query, FileHandler
from vianu.spock.settings import DATE_FORMAT

HEAD_FILE = Path(__file__).parent / "assets/head/scripts.html"
CSS_FILE = Path(__file__).parent / "assets/css/styles.css"

CONTAINER_TEMPLATE = """
<div class="card-container">
{cards}
</div>
"""

CARD_TEMPLATE = """
<div id="card_{nmbr}" class="card" onclick="clickHandler(this)" {data}>
  <div class="card-title">{title}</div>
  <div class="card-date">Date: {date}</div>
  <div class="card-sources">Sources: {sources}</div>
  <div class="card-stats">#Docs: {n_doc} | #ADR: {n_adr}</div>
</div>
"""

DETAILS_CONTAINER_TEMPLATE = """
<div id='details' class='details-container'>
  {inner_text}
</div>
"""

DETAILS_CONTAINER_INNER_TEXT_TEMPLATE = """
<div class='details-inner-title'>{title}</div>
<div class='details-inner-date'>{date}</div>
"""


NAMESPACE_KWARGS = {
    "source": ["pubmed"],
    "model": "ollama",
}


def _get_document_data(document: Document):
    data = []
    return ' '.join(data)


def _get_card_data(query: Query, data: List[Document]):
    return DETAILS_CONTAINER_INNER_TEXT_TEMPLATE.format(
        title=query.term,
        date=query.submission_date.strftime(DATE_FORMAT),
    )


def _format_search_card(card_nmbr: int, query: Query, data: List[Document]):
    title = query.term
    sources = ', '.join(query.sources)
    date = query.submission_date.strftime(DATE_FORMAT)
    n_doc = len(data)
    n_adr = sum([len(d.adverse_reactions) for d in data])
    data = f'data-details="{_get_card_data(query, data)}"'
    return CARD_TEMPLATE.format(nmbr=card_nmbr, data=data, title=title, date=date, sources=sources, n_doc=n_doc, n_adr=n_adr)


# Processing search input
SAMPLE_DATA = FileHandler('vianu/spock/assets/sample_data.json').read()
SAMPLE_QUERY_DAFALGAN = Query(term="dafalgan", sources=["pubmed", "ema"], submission_date=datetime.now())
SAMPLE_QUERY_SILDENAFIL = Query(term="sildenafil", sources=["pubmed", "ema"], submission_date=datetime.now())
SAMPLE_QUERY_FENTANYL = Query(term="fentanyl", sources=["pubmed", "ema"], submission_date=datetime.now())
SAMPLE_CARDS = [
    _format_search_card(0, SAMPLE_QUERY_DAFALGAN, SAMPLE_DATA),
    _format_search_card(1, SAMPLE_QUERY_SILDENAFIL, SAMPLE_DATA),
    _format_search_card(2, SAMPLE_QUERY_FENTANYL, SAMPLE_DATA),
]
def _process_pipeline(search_text):
    args = Namespace(term=search_text, **NAMESPACE_KWARGS)
    cards = SAMPLE_CARDS
    html_content = CONTAINER_TEMPLATE.format(cards="\n".join([c for c in cards]))

    return html_content


# Layout resembling the image
with gr.Blocks(head_paths=HEAD_FILE, css_paths=CSS_FILE, theme=gr.themes.Soft()) as demo:
    scraping_state = gr.State(value=False)
    ner_state = gr.State(value=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(value="vianu/spock/assets/images/spock_logo_circular.png", show_label=False, elem_id="logo-image")
        with gr.Column(scale=5):
            gr.Markdown("<div id='title-text'>SpoCK: Spotting Clinical Knowledge</div>")
                
    with gr.Row():
        search_input = gr.Textbox(label="Search", placeholder="Enter your search here...")
            
    with gr.Row():
        search_results = gr.HTML(label="Recently searched")
    
    with gr.Row():
        gr.HTML(DETAILS_CONTAINER_TEMPLATE.format(inner_text="details container placeholder"))

    search_input.submit(
        fn=_process_pipeline,
        inputs=search_input,
        outputs=search_results
    )   

if __name__ == '__main__':
    demo.launch()
