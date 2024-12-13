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
  <div class="title">{title}</div>
  <div class="info">Date: {date}</div>
  <div class="info">Sources: {sources}</div>
  <div class="info">#Docs: {n_doc} | #ADR: {n_adr}</div>
</div>
"""

DETAILS_CONTAINER = """
<div id='details' class='details-container'>
  details placeholder
</div>
"""

DETAILS_CONTAINER_CONTENT_TEMPLATE = """
<div class='items'>{items}</div>
"""

DETAILS_CONTAINER_ITEM_TEMPLATE = """
<div class='item'>
  <div class='top'>
    <div class='favicon'><img src='{favicon}' alt='Favicon'></div>
    <div class='title'><a href='{url}'>{title}</a></div>
  </div>
  <div class='bottom'>
    {text}
  </div>
</div>
"""

NER_MP_TEMPLATE = """
<span class='ner mp'>{text}|{class_}</span>
"""

NER_ADR_TEMPLATE = """
<span class='ner adr'>{text}|{class_}</span>
"""


NAMESPACE_KWARGS = {
    "source": ["pubmed"],
    "model": "ollama",
}


def _get_document_text(document: Document):
    text = f"<div>{document.text}</div>"
    for ne in document.medicinal_products:
        text = text.replace(
            ne.text, NER_MP_TEMPLATE.format(text=ne.text, class_=ne.class_)
        )
    for ne in document.adverse_reactions:
        text = text.replace(
            ne.text, NER_ADR_TEMPLATE.format(text=ne.text, class_=ne.class_)
        )
    return text


def _get_details_container_items(data: List[Document]):
    items = []
    max_title_lenth = 30
    for doc in data:
        items.append(
            DETAILS_CONTAINER_ITEM_TEMPLATE.format(
                favicon=doc.source_favicon_url,
                url=doc.url,
                title=doc.title[:max_title_lenth]
                + ("..." if len(doc.title) > max_title_lenth else ""),
                text=_get_document_text(doc),
                details="details",
            )
        )
    return "\n".join(items)


# TODO only save an id in data-id and then use that id to get the data from the data list
def _get_details_data(data: List[Document]):
    return DETAILS_CONTAINER_CONTENT_TEMPLATE.format(items=_get_details_container_items(data))


def _format_search_card(card_nmbr: int, query: Query, data: List[Document]):
    title = query.term
    sources = ", ".join(query.sources)
    date = query.submission_date.strftime(DATE_FORMAT)
    n_doc = len(data)
    n_adr = sum([len(d.adverse_reactions) for d in data])
    # TODO only save an id in data-id and then use that id to get the data from the data list
    data = f'data-details="{_get_details_data(data)}"'
    return CARD_TEMPLATE.format(
        nmbr=card_nmbr,
        data=data,
        title=title,
        date=date,
        sources=sources,
        n_doc=n_doc,
        n_adr=n_adr,
    )


# Processing search input
SAMPLE_DATA = FileHandler("vianu/spock/assets/sample_data.json").read()
SAMPLE_QUERY_DAFALGAN = Query(
    term="dafalgan", sources=["pubmed", "ema"], submission_date=datetime.now()
)
SAMPLE_QUERY_SILDENAFIL = Query(
    term="sildenafil", sources=["pubmed", "ema"], submission_date=datetime.now()
)
SAMPLE_QUERY_FENTANYL = Query(
    term="fentanyl", sources=["pubmed", "ema"], submission_date=datetime.now()
)


# TODO only save an id in data-id and then use that id to get the data from the data list
def _process_pipeline(search_text):
    args = Namespace(term=search_text, **NAMESPACE_KWARGS)
    formated_cards = [
        _format_search_card(0, SAMPLE_QUERY_DAFALGAN, SAMPLE_DATA),
        _format_search_card(1, SAMPLE_QUERY_SILDENAFIL, SAMPLE_DATA),
        # _format_search_card(2, SAMPLE_QUERY_FENTANYL, SAMPLE_DATA),
    ]
    html_content = CONTAINER_TEMPLATE.format(
        cards="\n".join([fc for fc in formated_cards])
    )
    return html_content


# Layout resembling the image
with gr.Blocks(
    head_paths=HEAD_FILE, css_paths=CSS_FILE, theme=gr.themes.Soft()
) as demo:
    scraping_state = gr.State(value=False)
    ner_state = gr.State(value=False)

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
        search_input = gr.Textbox(
            label="Search", placeholder="Enter your search here..."
        )

    with gr.Row():
        search_results = gr.HTML(label="Recently searched")

    with gr.Row():
        gr.HTML(DETAILS_CONTAINER)

    search_input.submit(
        fn=_process_pipeline, inputs=search_input, outputs=search_results
    )

if __name__ == "__main__":
    demo.launch()
