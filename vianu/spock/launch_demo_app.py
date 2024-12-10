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
<div class="card">
  <div class="card-title">{title}</div>
  <div class="card-date">Date: {date}</div>
  <div class="card-sources">Sources: {sources}</div>
  <div class="card-stats">#Docs: {n_doc} | #ADR: {n_adr}</div>
</div>
"""


NAMESPACE_KWARGS = {
    "source": ["pubmed"],
    "model": "ollama",
}

SAMPLE_DATA = FileHandler('vianu/spock/assets/sample_data.json').read()
SAMPLE_QUERY = Query(term="dafalgan", sources=["pubmed", "ema"], submission_date=datetime.now())


def _format_search_card(query: Query, data: List[Document]):
    title = query.term
    sources = ', '.join(query.sources)
    date = query.submission_date.strftime(DATE_FORMAT)
    n_doc = len(data)
    n_adr = sum([len(d.adverse_reactions) for d in data])
    return CARD_TEMPLATE.format(title=title, date=date, sources=sources, n_doc=n_doc, n_adr=n_adr)


# Processing search input
def _process_pipeline(search_text):
    args = Namespace(term=search_text, **NAMESPACE_KWARGS)
    
    cards = [_format_search_card(SAMPLE_QUERY, SAMPLE_DATA) for _ in range(5)]
    html_content = CONTAINER_TEMPLATE.format(cards="\n".join([c for c in cards]))

    return html_content


# Layout resembling the image
with gr.Blocks(head_paths=JS_FILE, css_paths=CSS_FILE, theme=gr.themes.Soft()) as demo:
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
            
    search_input.submit(
        fn=_process_pipeline,
        inputs=search_input,
        outputs=search_results
    )   

if __name__ == '__main__':
    demo.launch()
