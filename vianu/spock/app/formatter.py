import logging
from typing import List

from vianu.spock.src.base import Document, SpoCK
from vianu.spock.settings import DATE_FORMAT

logger = logging.getLogger(__name__)


JOBS_CONTAINER_CARD_TEMPLATE = """
<div class="card" onclick="cardClickHandler(this)">
  <div class="title">{title} {status}</div>
  <div class="info">Date: {date}</div>
  <div class="info">Model: {model}</div>
  <div class="info">Sources: {sources}</div>
  <div class="info">#docs: {n_doc} | #adr: {n_adr}</div>
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
    sorted_data = sorted(
        data,
        key=lambda x: (len(x.adverse_reactions), len(x.medicinal_products)),
        reverse=True,
    )
    items = _get_details_html_items(data=sorted_data)
    return DETAILS_CONTAINER_TEMPLATE.format(items=items)


def _get_status_html(status: str) -> str:
    """Get the HTML for the status."""
    if status == "running":
        return f"<span class='running'>({status.upper()})</span>"
    elif status == "completed":
        return f"<span class='completed'>({status.upper()})</span>"
    elif status == "stopped":
        return f"<span class='stopped'>({status.upper()})</span>"
    else:
        logger.error(f"unknown status: {status.upper()})")
        return "<span>(status unknown)</span>"


def get_job_card_html(card_nmbr: int, spock: SpoCK):
    """Get the HTML for the job card."""
    setup = spock.setup
    data = spock.data

    title = spock.setup.term
    status = _get_status_html(spock.status)
    date = setup.submission.strftime(DATE_FORMAT)
    model = setup.model
    sources = ", ".join(setup.source)
    n_doc = len(data)
    n_adr = sum([len(d.adverse_reactions) for d in data])
    return JOBS_CONTAINER_CARD_TEMPLATE.format(
        nmbr=card_nmbr,
        title=title,
        status=status,
        date=date,
        model=model,
        sources=sources,
        n_doc=n_doc,
        n_adr=n_adr,
    )
