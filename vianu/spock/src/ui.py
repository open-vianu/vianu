import logging
from typing import List

from vianu.spock.src.data_model import Document, Job
from vianu.spock.settings import DATE_FORMAT

logger = logging.getLogger(__name__)


JOBS_CONTAINER_CARD_TEMPLATE = """
<div class="card" onclick="cardClickHandler(this)">
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


def _get_details_html_items(data: List[Document]):
    """Get the HTML items for the details container. Each item contains the favicon, title, and the text with the 
    highlighted named entities.
    """
    items = []
    max_title_lenth = 120
    for doc in data:
        items.append(
            DETAILS_CONTAINER_ITEM_TEMPLATE.format(
                favicon=doc.source_favicon_url,
                url=doc.url,
                title=doc.title[:max_title_lenth]
                + ("..." if len(doc.title) > max_title_lenth else ""),
                text=doc.get_html(),
                details="details",
            )
        )
    return "\n".join(items)


def get_details_html(data: List[Document]):
    """Get the stacked HTML items for each document."""
    if len(data) == 0:
        return "<div>no results available (yet)</div>"
    return DETAILS_CONTAINER_TEMPLATE.format(items=_get_details_html_items(data=data))


def get_job_card_html(card_nmbr: int, job: Job, data: List[Document]):
    """Get the HTML for the job card."""
    title = job.term
    sources = ", ".join(job.source)
    date = job.submission.strftime(DATE_FORMAT)
    n_doc = len(data)
    n_adr = sum([len(d.adverse_reactions) for d in data])
    return JOBS_CONTAINER_CARD_TEMPLATE.format(
        nmbr=card_nmbr,
        title=title,
        date=date,
        sources=sources,
        n_doc=n_doc,
        n_adr=n_adr,
    )
