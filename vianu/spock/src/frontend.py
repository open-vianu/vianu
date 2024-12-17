import asyncio
import logging
from typing import List

from vianu.spock.src.data_model import Document, Job
from vianu.spock.settings import DATE_FORMAT

logger = logging.getLogger(__name__)

CARD_CONTAINER_TEMPLATE = """
<div class="jobs-container">
  {jobs}
</div>
"""

JOBS_CONTAINER_CARD_TEMPLATE = """
<div id="card_{nmbr}" class="card" onclick="clickHandler(this)" {data}>
  <div class="title">{title}</div>
  <div class="info">Date: {date}</div>
  <div class="info">Sources: {sources}</div>
  <div class="info">#Docs: {n_doc} | #ADR: {n_adr}</div>
</div>
"""

DETAILS_CONTAINER_TEMPLATE = """
<div id='details' class='details-container'>
  <div class='items'>{items}</div>
</div>
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
<span class='ner mp'>{text} | {class_}</span>
"""

NER_ADR_TEMPLATE = """
<span class='ner adr'>{text} | {class_}</span>
"""

def _get_document_text(doc: Document):
    text = f"<div>{doc.text}</div>"
    for ne in doc.medicinal_products:
        text = text.replace(
            ne.text, NER_MP_TEMPLATE.format(text=ne.text, class_=ne.class_)
        )
    for ne in doc.adverse_reactions:
        text = text.replace(
            ne.text, NER_ADR_TEMPLATE.format(text=ne.text, class_=ne.class_)
        )
    return text


def _get_details_container_items(data: List[Document]):
    items = []
    max_title_lenth = 50
    for doc in data:
        items.append(
            DETAILS_CONTAINER_ITEM_TEMPLATE.format(
                favicon=doc.source_favicon_url,
                url=doc.url,
                title=doc.title[:max_title_lenth]
                + ("..." if len(doc.title) > max_title_lenth else ""),
                text=_get_document_text(doc=doc),
                details="details",
            )
        )
    return "\n".join(items)


def get_details_of_data(data: List[Document]):
    return DETAILS_CONTAINER_TEMPLATE.format(items=_get_details_container_items(data=data))


def format_job_card(card_nmbr: int, job: Job, data: List[Document]):
    title = job.term
    sources = ", ".join(job.source)
    date = job.submission.strftime(DATE_FORMAT)
    n_doc = len(data)
    n_adr = sum([len(d.adverse_reactions) for d in data])
    # TODO only save an id in data-id and then use that id to get the data from the data list
    data = f'data-details="{get_details_of_data(data=data)}"'
    return JOBS_CONTAINER_CARD_TEMPLATE.format(
        nmbr=card_nmbr,
        data=data,
        title=title,
        date=date,
        sources=sources,
        n_doc=n_doc,
        n_adr=n_adr,
    )


async def conclusion(ner_queue: asyncio.Queue, orc_task: asyncio.Task):
    # Wait for orchestrator to finish and queue to be empty
    await orc_task
    await ner_queue.join()

