"""Module for scraping data from different sources.

The module contains three main classes:
- :class:`Scraper`: Abstract base class for scraping data from different sources
- :class:`PubmedScraper`: Class for scraping data from the PubMed database
- :class:`EMAScraper`: Class for scraping data from the European Medicines Agency
"""

from abc import ABC, abstractmethod
from argparse import Namespace
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
import logging
import re
from typing import List
import xml.etree.ElementTree as ET  # nosec

import aiohttp
from bs4 import BeautifulSoup
from bs4.element import Tag
import defusedxml.ElementTree as DET
import numpy as np
import pymupdf

from vianu.spock.src.base import Document, QueueItem  # noqa: F401
from vianu.spock.settings import MAX_CHUNK_SIZE, SCRAPING_SOURCES
from vianu.spock.settings import (
    PUBMED_ESEARCH_URL,
    PUBMED_DB,
    PUBMED_EFETCH_URL,
    PUBMED_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class Scraper(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def apply(self, args_: Namespace, queue_out: asyncio.Queue) -> None:
        """Main function for scraping data from a source.

        Args:
            - args_: the arguments for the spock pipeline
            - queue_out: the output queue for the scraped data
        """
        pass

    @staticmethod
    def split_text_into_chunks(
        text: str, chunk_size: int = MAX_CHUNK_SIZE, separator: str = " "
    ) -> List[str]:
        """Split a text into chunks of a given max size."""
        words = text.split(separator)
        N = len(words)
        s = min(chunk_size, N)
        n = N // s
        bnd = [round(i) for i in np.linspace(0, 1, n + 1) * N]

        chunks = [
            separator.join(words[start:stop]) for start, stop in zip(bnd[:-1], bnd[1:])
        ]
        return chunks

    @staticmethod
    async def _aiohttp_get_html(url: str, headers=None) -> str:
        """Get the content of a given URL by an aiohttp GET request."""
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url=url) as response:
                response.raise_for_status()
                text = await response.text()
        return text


@dataclass
class PubmedEntrezHistoryParams:
    """Class for optimizing Pubmed database retrieval for large numbers of documents.

    An example can be found here:
        https://www.ncbi.nlm.nih.gov/books/n/helpeutils/chapter3/#chapter3.Application_3_Retrieving_large
    """

    web: str
    key: str
    count: int


class PubmedScraper(Scraper):
    """Class for scraping data from the PubMed database.

    The scraper uses the Pubmed API to search for relevant documents. From the list of results it creates a list of
    :class:`Document` objects by the following main steps:
        - Extract all PubmedArticle elements from the search results (other types are ignored)
        - Extract the AbstractText from the PubmedArticle (if there is no abstract, the document is ignored)
    """

    _source = "pubmed"
    _source_url = "https://pubmed.ncbi.nlm.nih.gov/"
    _source_favicon_url = "https://www.ncbi.nlm.nih.gov/favicon.ico"

    _robots_txt_url = "https://www.ncbi.nlm.nih.gov/robots.txt"

    @staticmethod
    def _get_entrez_history_params(text: str) -> PubmedEntrezHistoryParams:
        """Retrieving the entrez history parameters for optimized search when requesting large numbers of documents.
        An example can be found here:
            https://www.ncbi.nlm.nih.gov/books/n/helpeutils/chapter3/#chapter3.Application_3_Retrieving_large
        """
        web = re.search(r"<WebEnv>(\S+)</WebEnv>", text).group(1)
        key = re.search(r"<QueryKey>(\d+)</QueryKey>", text).group(1)
        count = int(re.search(r"<Count>(\d+)</Count>", text).group(1))
        return PubmedEntrezHistoryParams(web=web, key=key, count=count)

    async def _pubmed_esearch(self, term: str) -> str:
        """Search the Pubmed database with a given term and POST the results to entrez history server."""
        url = f"{PUBMED_ESEARCH_URL}?db={PUBMED_DB}&term={term}&usehistory=y"
        self.logger.debug(f"search pubmed database with url={url}")
        esearch = await self._aiohttp_get_html(url=url)
        return esearch

    async def _pubmed_efetch(
        self, params: PubmedEntrezHistoryParams, max_docs_src: int
    ) -> List[str]:
        """Retrieve the relevant documents from the entrez history server."""
        # Reduce the number of documents to be retrieved for efficiency
        N = min(max_docs_src, int(params.count))
        if N < params.count:
            self.logger.warning(
                f"from the total number of documents={params.count} only {N} will be retrieved"
            )

        # Iterate over the batches of documents (with fixed batch size)
        batch_size = min(params.count, PUBMED_BATCH_SIZE)
        self.logger.debug(
            f"fetch #docs={N} in {N // batch_size + 1} batch(es) of size <= {batch_size}"
        )
        batches = []
        for retstart in range(0, N, batch_size):
            # Prepare URL for retrieving next batch of documents but stop if the maximum number is reached
            retmax = min(max_docs_src - len(batches) * batch_size, batch_size)
            url = f"{PUBMED_EFETCH_URL}?db={PUBMED_DB}&WebEnv={params.web}&query_key={params.key}&retstart={retstart}&retmax={retmax}"
            self.logger.debug(f"fetch documents with url={url}")

            # Fetch the documents
            efetch = await self._aiohttp_get_html(url=url)
            batches.append(efetch)
        return batches

    def _extract_medline_citation(self, element: ET.Element) -> ET.Element | None:
        """Extract the MedlineCitation element from a PubmedArticle element."""
        # Find and extract the MedlineCitation element
        citation = element.find("MedlineCitation")
        if citation is None:
            self.logger.warning('no "MedlineCitation" element found')
            return None
        return citation

    @staticmethod
    def _extract_pmid(element: ET.Element) -> str | None:
        """Extract the PMID from a MedlineCitation element."""
        pmid = element.find("PMID")
        return pmid.text if pmid is not None else None

    @staticmethod
    def _extract_article(element: ET.Element) -> ET.Element | None:
        """Extract the article element from a PubmedArticle element."""
        # Find and extract the Article element
        article = element.find("Article")
        return article

    @staticmethod
    def _extract_title(article: ET.Element) -> str | None:
        """Extract the title from an Article element."""
        title = article.find("ArticleTitle")
        return title.text if title is not None else None

    @staticmethod
    def _extract_abstract(article: ET.Element) -> str | None:
        """Extract the abstract from an Article element."""
        separator = "\n\n"
        abstract = article.find("Abstract")
        if abstract is not None:
            abstract = separator.join(
                [a.text for a in abstract.findall("AbstractText") if a.text is not None]
            )
        return abstract

    @staticmethod
    def _extract_language(article: ET.Element) -> str | None:
        """Extract the language from an Article element."""
        language = article.find("Language")
        return language.text if language is not None else None

    @staticmethod
    def _extract_date(article: ET.Element) -> datetime | None:
        """Extract the publication date from an Article element."""
        date = article.find("ArticleDate")
        if date is None:
            return None
        year = int(date.find("Year").text)
        month = int(date.find("Month").text)
        day = int(date.find("Day").text)
        return datetime(year=year, month=month, day=day)

    @staticmethod
    def _extract_publication_types(article: ET.Element) -> List[str]:
        """Extract the publication types from an Article element."""
        return [
            t.text
            for t in article.find("PublicationTypeList").findall("PublicationType")
        ]

    def _parse_pubmed_articles(self, batches: List[str]) -> List[Document]:
        """Parse batches of ET.Elements into a single list of Document objects"""
        data = []
        for ib, text in enumerate(batches):
            pubmed_articles = DET.fromstring(text).findall("PubmedArticle")
            self.logger.debug(f"found #articles={len(pubmed_articles)} in batch {ib}")
            for ie, element in enumerate(pubmed_articles):
                self.logger.debug(f"parsing PubmedArticle {ie} of batch {ib}")
                # Extract MedlineCitation and its PMID from PubmedArticle
                citation = self._extract_medline_citation(element=element)
                if citation is None:
                    self.logger.debug(
                        f"no citation found in PubmedArticle {ie} of batch {ib}"
                    )
                    continue
                pmid = self._extract_pmid(element=citation)

                # Extract the Article element from the PubmedArticle
                article = self._extract_article(element=citation)
                if article is None:
                    self.logger.debug(
                        f"no article found in PubmedArticle {ie} of batch {ib}"
                    )
                    continue

                # Extract the relevant information from the Article element
                title = self._extract_title(article=article)
                text = self._extract_abstract(article=article)
                if text is None:
                    self.logger.debug(
                        f"no abstract found in PubmedArticle {ie} of batch {ib}"
                    )
                    continue
                language = self._extract_language(article=article)
                publication_date = self._extract_date(article=article)

                # Split long texts into chunks
                texts = self.split_text_into_chunks(text=text)

                # Create the Document object(s)
                for txt in texts:
                    document = Document(
                        id_=f"{self._source_url} {title} {txt} {language} {publication_date}",
                        text=txt,
                        source=self._source,
                        title=title,
                        url=f"{self._source_url}{pmid}/",
                        source_url=self._source_url,
                        source_favicon_url=self._source_favicon_url,
                        language=language,
                        publication_date=publication_date,
                    )
                    data.append(document)
        self.logger.debug(f"parsed #docs={len(data)} from #batches={len(batches)}")
        return data

    async def apply(self, args_: Namespace, queue_out: asyncio.Queue) -> None:
        """Query and retrieve all PubmedArticle Documents for the given search term.

        The retrieval is using two main functionalities of the Pubmed API:
        - ESearch: Identify the relevant documents and store them in the entrez history server
        - EFetch: Retrieve the relevant documents from the entrez history server

        Args:
            - args_: the arguments for the spock pipeline
            - queue_out: the output queue for the scraped data
        """
        term = args_.term
        max_docs_src = args_.max_docs_src
        self.logger.debug(
            f"starting scraping the source={self._source} with term={term}"
        )

        # Search for relevant documents with a given term
        esearch = await self._pubmed_esearch(term=term)

        # Retrieve relevant documents in batches
        params = self._get_entrez_history_params(esearch)
        batches = await self._pubmed_efetch(params=params, max_docs_src=max_docs_src)

        # Parse documents from batches
        documents = self._parse_pubmed_articles(batches=batches)
        documents = documents[:max_docs_src]

        # Add documents to the queue
        for i, doc in enumerate(documents):
            id_ = f"{self._source}_{i}"
            item = QueueItem(id_=id_, doc=doc)
            await queue_out.put(item)

        self.logger.info(
            f"retrieved #docs={len(documents)} in source={self._source} for term={term}"
        )


@dataclass
class SearchResults:
    """Class for storing the search results from different databases."""

    count: int | None = None
    n_pages: int | None = None
    items: List[Tag] = field(default_factory=list)


class EMAScraper(Scraper):
    """Class for scraping data from the European Medicines Agency.

    The scraper uses the same API as the web interface of the EMA to search for relevant documents. From the list of
    results it creates a list of :class:`Document` objects by the following main steps:
        - Search the EMA database (filter for PDF documents only)
        - Extract the text from the PDF documents
        - Use regex to find texts where adverse drug reactions (or similar) are mentioned
        - Return the most recent :class:`Document` objects
    """

    _source = "ema"
    _source_url = "https://www.ema.europa.eu"
    _source_favicon_url = (
        "https://www.ema.europa.eu/themes/custom/ema_theme/favicon.ico"
    )

    _pdf_search_template = (
        "https://www.ema.europa.eu/en/search?search_api_fulltext={term}"
        "&f%5B0%5D=ema_search_custom_entity_bundle%3Adocument"  # This part is added to only retrieve PDF documents
        "&f%5B1%5D=ema_search_entity_is_document%3ADocument"
    )
    _headers = {
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Host": "www.ema.europa.eu",
    }

    _robots_txt_url = "https://www.ema.europa.eu/robots.txt"

    def _extract_search_results_count(self, soup: BeautifulSoup) -> int | None:
        """Extract the number of search results."""
        span = soup.find("span", class_="source-summary-count")
        if span is None:
            self.logger.warning("no search results count found")
            return None
        return int(span.text.strip("()"))

    def _extract_number_of_pages(self, soup: BeautifulSoup) -> int | None:
        """Extract the number of pages from the search results."""
        nav = soup.find("nav", class_="pager")
        if nav is not None:
            a = nav.find("a", {"class": "page-link", "aria-label": "Last"})
            if a and a.has_attr("href"):
                href = a["href"]
                match = re.search(r"&page=(\d+)", href)
                if match:
                    return int(match.group(1)) + 1
        self.logger.warning("no pager found")
        return None

    @staticmethod
    def _extract_search_item_divs(soup: BeautifulSoup) -> List[Tag]:
        """Extract the list of div elements contining the different search results."""
        parent = soup.find("div", class_=["row", "row-cols-1"])
        return parent.find_all("div", class_="col")

    async def _ema_document_search(self, term: str, max_docs_src: int) -> SearchResults:
        """Search the EMA database for PDF documents with a given term."""

        # Get initial search results
        url = self._pdf_search_template.format(term=term)
        self.logger.debug(f"search ema database with url={url}")
        content = await self._aiohttp_get_html(url=url, headers=self._headers)
        soup = BeautifulSoup(content, "html.parser")

        # Get the number of search results and number of pages
        count = self._extract_search_results_count(soup=soup)
        n_pages = self._extract_number_of_pages(soup=soup)

        # Extract the divs containing the search results
        items = []
        if count is not None and count > 0:
            # Extract items from page=0
            items_from_page = self._extract_search_item_divs(soup=soup)
            items.extend(items_from_page)

            # Extract items from page=1, 2, ...
            if n_pages is not None and n_pages > 1:
                for i in range(1, n_pages):
                    url = f"{url}&page={i}"
                    content = await self._aiohttp_get_html(
                        url=url, headers=self._headers
                    )
                    soup = BeautifulSoup(content, "html.parser")

                    items_from_page = self._extract_search_item_divs(soup=soup)
                    items.extend(items_from_page)

                    if len(items) >= max_docs_src:
                        self.logger.debug(
                            f"found #items={len(items)} in #pages={i + 1}"
                        )
                        break

            # Check for extraction mismatch
            if len(items) != count:
                self.logger.warning(
                    f"mismatch #items={len(items)} and the total count={count}"
                )

        self.logger.debug(f"extracted #items={len(items)} in #pages={n_pages}")
        return SearchResults(count=count, n_pages=n_pages, items=items)

    @staticmethod
    def _extract_title(tag: Tag) -> str | None:
        """Extract the title of the document."""
        title = tag.find("p", class_="file-title")
        return title.text if title is not None else None

    def _extract_url(self, tag: Tag) -> str | None:
        """Extract the links href to the relevant PDF document."""
        link = tag.find("a", href=True)
        if link is None:
            self.logger.warning("no link found")
            return None

        href = link["href"]
        url = f"{self._source_url}{href}" if href.startswith("/") else href
        if not url.endswith(".pdf"):
            self.logger.warning(f"url={url} does not point to a PDF document")
            return None
        return url

    async def _extract_text(self, url: str) -> str:
        """Extract the text from the PDF document."""
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()  # Read the entire content

                # Create a BytesIO object from the content
                pdf_stream = BytesIO(content)

                # Open the PDF with PyMuPDF using the BytesIO object
                doc = pymupdf.open(stream=pdf_stream, filetype="pdf")

                # Extract text from all pages
                text = "\n".join([page.get_text() for page in doc])

                # Close the document
                doc.close()
        return text

    @staticmethod
    def _extract_language(tag: Tag) -> str | None:
        """Extract the language of the document."""
        lang_tag = tag.find("p", class_="language-meta")
        if lang_tag is None:
            return None
        text = lang_tag.text
        start = text.find("(")
        stop = text.find(")")
        if start != -1 and stop != -1:
            return text[start + 1 : stop].lower()
        return None

    @staticmethod
    def _extract_date(tag: Tag) -> datetime | None:
        """Extract the publication date of the document."""
        time_tag = tag.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            return datetime.fromisoformat(time_tag["datetime"])
        return None

    async def _parse_items(self, items: List[Tag]) -> List[Document]:
        """From a list of divs containing the search results, extract the relevant information and parse it into a list
        of :class:`Document` objects.
        """
        data = []
        for i, tag in enumerate(items):
            url = self._extract_url(tag=tag)
            if url is None:
                self.logger.debug(f"no url found for item {i}")
                continue

            # Extract the relevant information from the document
            self.logger.debug(f"parsing document with url={url}")
            title = self._extract_title(tag=tag)
            text = await self._extract_text(url=url)
            language = self._extract_language(tag=tag)
            publication_date = self._extract_date(tag=tag)

            # Split long texts into chunks
            texts = self.split_text_into_chunks(text=text)

            # Create the Document object(s)
            for text in texts:
                document = Document(
                    id_=f"{self._source_url} {title} {text} {language} {publication_date}",
                    text=text,
                    source=self._source,
                    title=title,
                    url=url,
                    source_url=self._source_url,
                    source_favicon_url=self._source_favicon_url,
                    language=language,
                    publication_date=publication_date,
                )
                data.append(document)
        self.logger.debug(f"created #docs={len(data)}")
        return data

    async def apply(self, args_: Namespace, queue_out: asyncio.Queue) -> None:
        """Query and retrieve all PRAC documents for the given search term.

        Args:
            - args_: the arguments for the spock pipeline
            - queue_out: the output queue for the scraped data
        """
        term = args_.term
        max_docs_src = args_.max_docs_src
        self.logger.debug(
            f"starting scraping the source={self._source} with term={term}"
        )

        # Search for relevant documents with a given term
        search_results = await self._ema_document_search(
            term=term, max_docs_src=max_docs_src
        )
        n_items = len(search_results.items)
        if n_items > max_docs_src:
            self.logger.warning(
                f"from #items={n_items} only max_docs_src={max_docs_src} will be parsed"
            )
        items = search_results.items[:max_docs_src]

        # Parse the documents
        data = await self._parse_items(items=items)
        n_data = len(data)
        if n_data > max_docs_src:
            self.logger.warning(
                f"the #items={n_items} were chunked into #documents={n_data} from where only max_docs_src={max_docs_src} will be added to the queue"
            )
        data = data[:max_docs_src]

        # Add documents to the queue
        for i, doc in enumerate(data):
            id_ = f"{self._source}_{i}"
            item = QueueItem(id_=id_, doc=doc)
            await queue_out.put(item)

        self.logger.info(
            f"retrieved #docs={len(data)} in source={self._source} for term={term}"
        )


class MHRAScraper(Scraper):
    """Class for scraping data from the Medicines and Healthcare products Regulatory Agency.

    The scraper the MHRAs **Drug Safety Update** search API for retrieving relevant documents. From the list of results
    it creates a list of :class:`Document` objects.
    """

    _source = "mhra"
    _source_url = "https://www.gov.uk/durg-safety-update"
    _source_favicon_url = "https://www.gov.uk/favicon.ico"

    _search_template = "https://www.gov.uk/drug-safety-update?keywords={term}"
    _source_base_url = "https://www.gov.uk"
    _language = "en"

    def _extract_search_results_count(self, parent: Tag) -> int | None:
        """Extract the number of search results."""
        div = parent.find("div", class_="result-info__header")
        h2 = div.find("h2") if div else None

        if h2 is None:
            self.logger.warning("no search results count found")
            return None
        text = h2.get_text(strip=True)
        count = int(re.search(r"\d+", text).group())
        return count

    @staticmethod
    def _extract_search_item_divs(parent: Tag) -> List[Tag]:
        """Extract the divs containing the search results."""
        return parent.find_all("li", class_="gem-c-document-list__item")

    async def _mhra_document_search(self, term: str) -> SearchResults:
        """Search the MHRA database for documents with a given term."""

        # Get search results and extract divs containing the search results
        url = self._search_template.format(term=term)
        self.logger.debug(f"search mhra's drug safety update database with url={url}")
        content = await self._aiohttp_get_html(url=url)
        soup = BeautifulSoup(content, "html.parser")
        parent = soup.find(
            "div",
            class_=[
                "govuk-grid-column-two-thirds",
                "js-live-search-results-block",
                "filtered-results",
            ],
        )

        # Extract the number of search results
        count = self._extract_search_results_count(parent=parent)

        # Extract the divs containing the search results
        items = []
        if count is not None and count > 0:
            items = self._extract_search_item_divs(
                parent=parent
            )  # For a given search term, the site shows all the results without pagination.

            # Check for extraction mismatch
            if len(items) != count:
                self.logger.warning(
                    f"mismatch #items={len(items)} and the total count={count}"
                )

        self.logger.debug(f"found #items={len(items)}")
        return SearchResults(count=count, items=items)

    def _extract_url(self, link: Tag) -> str | None:
        """Extract the url to the document."""
        href = link["href"]
        return f"{self._source_base_url}{href}" if href.startswith("/") else href

    async def _extract_text(self, url: str) -> str:
        """Extract the text from the document."""
        content = await self._aiohttp_get_html(url=url)
        soup = BeautifulSoup(content, "html.parser")
        main = soup.find("main")
        text = main.get_text()

        # Clean the spaces and newlines
        text = re.sub(r"(\n\s*){3,}", "\n\n", text)
        text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _extract_date(tag: Tag) -> datetime | None:
        """Extract the publication date of the document."""
        time_tag = tag.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            return datetime.fromisoformat(time_tag["datetime"])
        return None

    async def _parse_items(self, items: List[Tag]) -> List[Document]:
        """From a list of divs containing the search results (items), extract the relevant information and parse it into a list
        of :class:`Document` objects.
        """
        data = []
        for i, tag in enumerate(items):
            link = tag.find("a", href=True)
            if link is None:
                self.logger.warning(f"no link found for item {i}")
                continue
            url = self._extract_url(link=link)

            # Extract the relevant information from the document
            self.logger.debug(f"parsing item with url={url}")
            title = link.get_text(strip=True)
            text = await self._extract_text(url=url)
            publication_date = self._extract_date(tag=tag)

            # Split long texts into chunks
            texts = self.split_text_into_chunks(text=text)

            # Create the Document object(s)
            for text in texts:
                document = Document(
                    id_=f"{self._source_url} {title} {text} {publication_date}",
                    text=text,
                    source=self._source,
                    title=title,
                    url=url,
                    source_url=self._source_url,
                    source_favicon_url=self._source_favicon_url,
                    language=self._language,
                    publication_date=publication_date,
                )
                data.append(document)
        self.logger.debug(f"created #docs={len(data)}")
        return data

    async def apply(self, args_: Namespace, queue_out: asyncio.Queue) -> None:
        """Query and retrieve all drug safety updates for the given search term.

        Args:
            - args_: the arguments for the spock pipeline
            - queue_out: the output queue for the scraped data
        """
        term = args_.term
        max_docs_src = args_.max_docs_src
        self.logger.debug(
            f"starting scraping the source={self._source} with term={term}"
        )

        # Search for relevant documents with a given term
        search_results = await self._mhra_document_search(term=term)
        n_items = len(search_results.items)
        if n_items > max_docs_src:
            self.logger.warning(
                f"from #items={n_items} only max_docs_src={max_docs_src} will be parsed"
            )
        items = search_results.items[:max_docs_src]

        # Parse the documents
        data = await self._parse_items(items=items)
        n_data = len(data)
        if n_data > max_docs_src:
            self.logger.warning(
                f"the #items={n_items} were chunked into #documents={n_data} from where only max_docs_src={max_docs_src} will be added to the queue"
            )
        data = data[:max_docs_src]

        # Add documents to the queue
        for i, doc in enumerate(data):
            id_ = f"{self._source}_{i}"
            item = QueueItem(id_=id_, doc=doc)
            await queue_out.put(item)

        self.logger.info(
            f"retrieved #docs={len(data)} in source={self._source} for term={term}"
        )


class FDAScraper(Scraper):
    """Class for scraping data from the Food and Drug Administration.

    The scraper uses the  the same API as the web interface fo the FDA to search for relevant documents. By default the search applies the following filter:
        - sorting by highest relevance
        - filter for results from the Center of Drug Evaluation and Research
        - filter for English language
        - filter for Drugs
    """

    _source = "fda"
    _source_url = "https://www.fda.gov"
    _source_favicon_url = "https://www.fda.gov/favicon.ico"

    _search_template = (
        "https://www.fda.gov/search?s={term}"
        "&items_per_page=10"
        "&sort_bef_combine=rel_DESC"  # Sort by relevance
        "&f%5B0%5D=center%3A815"  # Filter for the Center for Drug Evaluation and Research
        "&f%5B1%5D=language%3A1404"  # Filter for English language
        "&f%5B2%5D=prod%3A2312"  # Filter for the Drugs section
    )
    _language = "en"

    def _extract_search_results_count(self, soup: BeautifulSoup) -> int | None:
        """Extract the number of search results from the search info section."""
        parent = soup.find("div", class_="lcds-search-filters__info")
        if parent is not None:
            div = parent.find("div", class_="view-header")
            match = re.search(r"of (\d+) entr[y|ies]", div.text)
            if match:
                return int(match.group(1))
        self.logger.warning("no search info found")
        return None

    def _extract_number_of_pages(self, soup: BeautifulSoup) -> int | None:
        """Extract the number of pages from the search results."""
        nav = soup.find("nav", class_=["pager-nav", "text-center"])
        if nav is not None:
            last_page = nav.find("li", class_=["pager__item", "pager__item--last"])
            a = last_page.find("a")
            if a and a.has_attr("href"):
                href = a["href"]
                match = re.search(r"&page=(\d+)", href)
                if match:
                    return int(match.group(1)) + 1
        self.logger.warning("no pager found")
        return None

    @staticmethod
    def _extract_search_item_divs(soup: BeautifulSoup) -> List[Tag]:
        """Extract the divs containing the search results."""
        parent = soup.find("div", class_="view-content")
        return parent.find_all("div", recursive=False)

    async def _fda_document_search(self, term: str, max_docs_src: int) -> SearchResults:
        """Search the FDA database for documents with a given term."""

        # Get search results
        url = self._search_template.format(term=term)
        self.logger.debug(f"search fda database with url={url}")
        content = await self._aiohttp_get_html(url=url)
        soup = BeautifulSoup(content, "html.parser")

        # Get the number of search results and the number of pages
        count = self._extract_search_results_count(soup=soup)
        n_pages = self._extract_number_of_pages(soup=soup)

        # Extract the divs containing the search results
        items = []
        if count is not None and count > 0:
            # Extract items from page=0
            items_from_page = self._extract_search_item_divs(soup=soup)
            items.extend(items_from_page)

            # Extract items from page=1, 2, ...
            if n_pages is not None and n_pages > 1:
                for i in range(1, n_pages):
                    url = f"{url}&page={i}"
                    content = await self._aiohttp_get_html(url=url)
                    soup = BeautifulSoup(content, "html.parser")

                    items_from_page = self._extract_search_item_divs(soup=soup)
                    items.extend(items_from_page)

                    if len(items) >= max_docs_src:
                        self.logger.debug(
                            f"found #items={len(items)} in #pages={i + 1}"
                        )
                        break

            # Check for extraction mismatch
            if len(items) != count:
                self.logger.warning(
                    f"mismatch #items={len(items)} and the total count={count}"
                )

        self.logger.debug(f"extracted #items={len(items)} in #pages={n_pages}")
        return SearchResults(count=count, n_pages=n_pages, items=items)

    @staticmethod
    def _extract_title(main: Tag) -> str | None:
        """Extract the title of the document."""
        header = main.find("header", class_=["row", "content-header"])
        if header is not None:
            h1 = header.find("h1", class_=["content-title", "text-center"])
            if h1 is not None:
                return h1.text
        return None

    @staticmethod
    def _extract_text(main: Tag) -> str | None:
        """Extract the text from the document."""
        body = main.find(
            "div", attrs={"class": "col-md-8 col-md-push-2", "role": "main"}
        )
        return body.get_text() if body is not None else None

    @staticmethod
    def _extract_date(main: Tag) -> datetime | None:
        """Extract the publication date of the document."""
        dl = main.find("dl", class_="lcds-description-list--grid")
        if dl is not None:
            dd = dl.find("dd", class_="cell-2_2")
            time_tag = dd.find("time")
        else:
            time_tag = main.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            return datetime.fromisoformat(time_tag["datetime"])
        return None

    async def _parse_item_page(self, url: str) -> List[Document]:
        content = await self._aiohttp_get_html(url=url)
        soup = BeautifulSoup(content, "html.parser")
        main = soup.find("main")

        # Extract the relevant information from the document
        title = self._extract_title(main=main)
        text = self._extract_text(main=main)
        publication_date = self._extract_date(main=main)

        # Split long texts into chunks
        texts = self.split_text_into_chunks(text=text)

        # Create the Document object(s)
        data = []
        for text in texts:
            document = Document(
                id_=f"{self._source_url} {title} {text} {publication_date}",
                text=text,
                source=self._source,
                title=title,
                url=url,
                source_url=self._source_url,
                source_favicon_url=self._source_favicon_url,
                language=self._language,
                publication_date=publication_date,
            )
            data.append(document)
        return data

    async def _parse_items(self, items: List[Tag]) -> List[Document]:
        """From a list of divs containing the search results, extract the relevant information and parse it into a list
        of :class:`Document` objects.
        """
        data = []
        for i, tag in enumerate(items):
            link = tag.find("a", href=True)
            if link is None:
                self.logger.warning(f"no link found for item {i}")
                continue
            url = self._source_url + link["href"]

            # Create the Document object
            self.logger.debug(f"parsing item with url={url}")
            page_data = await self._parse_item_page(url=url)
            data.extend(page_data)
        self.logger.debug(f"created #docs={len(data)}")
        return data

    async def apply(self, args_: Namespace, queue_out: asyncio.Queue) -> None:
        term = args_.term
        max_docs_src = args_.max_docs_src
        self.logger.debug(
            f"starting scraping the source={self._source} with term={term}"
        )

        # Search for relevant documents with a given term
        search_results = await self._fda_document_search(
            term=term, max_docs_src=max_docs_src
        )
        n_items = len(search_results.items)
        if n_items > max_docs_src:
            self.logger.warning(
                f"from #items={n_items} only max_docs_src={max_docs_src} will be parsed"
            )
        items = search_results.items[:max_docs_src]

        # Parse the documents
        data = await self._parse_items(items=items)
        n_data = len(data)
        if n_data > max_docs_src:
            self.logger.warning(
                f"the #items={n_items} were chunked into #documents={n_data} from where only max_docs_src={max_docs_src} will be added to the queue"
            )
        data = data[:max_docs_src]

        # Add documents to the queue
        for i, doc in enumerate(data):
            id_ = f"{self._source}_{i}"
            item = QueueItem(id_=id_, doc=doc)
            await queue_out.put(item)

        self.logger.info(
            f"retrieved #docs={len(data)} in source={self._source} for term={term}"
        )


_SCRAPERS = [PubmedScraper, EMAScraper, MHRAScraper, FDAScraper]
_SOURCE_TO_SCRAPER = {src: scr for src, scr in zip(SCRAPING_SOURCES, _SCRAPERS)}
if not len(_SCRAPERS) == len(SCRAPING_SOURCES):
    raise ValueError("number of scrapers and sources do not match")


async def _scraping(
    args_: Namespace, queue_in: asyncio.Queue, queue_out: asyncio.Queue
) -> None:
    """Pop a source (str) from the input queue, perform  the scraping task with the given term, and put the results in
    the output queue until the input queue is empty.

    Args:
        - args_: the arguments for the spock pipeline
        - queue_in: the input queue containing the sources to scrape
        - queue_out: the output queue for the scraped data
    """

    while True:
        # Get source from input queue
        source = await queue_in.get()

        # Check stopping condition
        if source is None:
            queue_in.task_done()
            break

        # Get the scraper and apply it to the term
        scraper = _SOURCE_TO_SCRAPER.get(source)  # type: type[Scraper]
        if scraper is None:
            logger.error(f"unknown source={source}")
            queue_in.task_done()
            break

        try:
            await scraper().apply(args_=args_, queue_out=queue_out)
        except Exception as e:
            logger.error(
                f"error during scraping for source={source} and term={args_.term}: {e}"
            )
            queue_in.task_done()
            continue
        queue_in.task_done()


def create_tasks(
    args_: Namespace, queue_in: asyncio.Queue, queue_out: asyncio.Queue
) -> List[asyncio.Task]:
    """Create the asyncio scraping tasks."""
    n_tasks = args_.n_scp_tasks
    logger.info(f"setting up {n_tasks} scraping task(s) for source(s)={args_.source}")
    tasks = [
        asyncio.create_task(
            _scraping(args_=args_, queue_in=queue_in, queue_out=queue_out)
        )
        for _ in range(n_tasks)
    ]
    return tasks
