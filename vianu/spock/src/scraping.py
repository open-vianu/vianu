"""Module for scraping data from different sources.

The module contains three main classes:
- :class:`Scraper`: Abstract base class for scraping data from different sources
- :class:`PubmedScraper`: Class for scraping data from the PubMed database
- :class:`EMAScraper`: Class for scraping data from the European Medicines Agency
"""

from abc import ABC, abstractmethod
import aiohttp
from argparse import Namespace
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np
import re
from typing import List
import xml.etree.ElementTree as ET

from .data_model import Document, QueueItem
from ..settings import SCRAPING_SOURCES, MAX_CHUNK_SIZE
from ..settings import PUBMED_ESEARCH_URL, PUBMED_DB, PUBMED_EFETCH_URL, PUBMED_BATCH_SIZE

logger = logging.getLogger(__name__)



class Scraper(ABC):

    _word_separator = ' '

    @abstractmethod
    async def apply(self, term: str, queue: asyncio.Queue) -> List[Document]:
        """Main function for scraping data from a source."""
        pass


    def split_text_into_chunks(self, text: str, max_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """Split a text into chunks of a given max size."""
        words = text.split(self._word_separator)
        N = len(words)
        s = min(max_size, N)
        n = N // s
        bnd = [round(i) for i in np.linspace(0, 1, n+1) * N]

        chunks = [self._word_separator.join(words[start:stop]) for start, stop in zip(bnd[:-1], bnd[1:])]
        return chunks

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

    _source = 'PubMed'
    _source_url = 'https://pubmed.ncbi.nlm.nih.gov/'
    _source_favicon_url = 'https://www.ncbi.nlm.nih.gov/favicon.ico'

    @staticmethod
    def _get_entrez_history_params(text: str) -> PubmedEntrezHistoryParams:
        """Retrieving the entrez history parameters for optimized search when requesting large numbers of documents.
        An example can be found here:
            https://www.ncbi.nlm.nih.gov/books/n/helpeutils/chapter3/#chapter3.Application_3_Retrieving_large
        """
        web = re.search(r'<WebEnv>(\S+)<\/WebEnv>', text).group(1)
        key = re.search(r'<QueryKey>(\d+)<\/QueryKey>', text).group(1)
        count = int(re.search(r'<Count>(\d+)<\/Count>', text).group(1))
        return PubmedEntrezHistoryParams(web=web, key=key, count=count)
    
    @staticmethod
    async def _pubmed_esearch(term: str) -> str:
        """Search the Pubmed database with a given term and POST the results to entrez history server."""
        url = f'{PUBMED_ESEARCH_URL}?db={PUBMED_DB}&term={term}&usehistory=y'
        logger.debug(f'search pubmed database with url={url}')
        async with aiohttp.ClientSession() as session:
            async with session.get(url=url) as response:
                response.raise_for_status()
                esearch = await response.text()
        return esearch

    @staticmethod
    async def _pubmed_efetch(params: PubmedEntrezHistoryParams) -> List[str]:
        """Retrieve the relevant documents from the entrez history server."""
        logger.debug(f'fetch #docs={params.count} in {params.count // PUBMED_BATCH_SIZE + 1} batch(es) of size <= {PUBMED_BATCH_SIZE}')
        batches = []
        for retstart in range(0, int(params.count), PUBMED_BATCH_SIZE):
            url = f'{PUBMED_EFETCH_URL}?db={PUBMED_DB}&WebEnv={params.web}&query_key={params.key}&retstart={retstart}&retmax={PUBMED_BATCH_SIZE}'
            logger.debug(f'fetch documents with url={url}')
            async with aiohttp.ClientSession() as session:
                async with session.get(url=url) as response:
                    response.raise_for_status()
                    efetch = await response.text()
            batches.append(efetch)
        return batches

    @staticmethod
    def _extract_medline_citation(element: ET.Element) -> ET.Element | None:
        """Extract the MedlineCitation element from a PubmedArticle element."""
        # Find and extract the MedlineCitation element
        citation = element.find('MedlineCitation')
        if citation is None:
            logger.warning('no "MedlineCitation" element found')
            return None
        return citation

    @staticmethod
    def _extract_pmid(element: ET.Element) -> str | None:
        """Extract the PMID from a MedlineCitation element."""
        pmid = element.find('PMID')
        return pmid.text if pmid is not None else None

    @staticmethod
    def _extract_article(element: ET.Element) -> ET.Element | None:
        """Extract the article element from a PubmedArticle element."""
        # Find and extract the Article element
        article = element.find('Article')
        if article is None:
            logger.warning('no "Article" element found')
        return article

    @staticmethod
    def _extract_title(article: ET.Element) -> str | None:
        """Extract the title from an Article element."""
        title = article.find('ArticleTitle')
        return title.text if title is not None else None

    @staticmethod
    def _extract_abstract(article: ET.Element) -> str | None:
        """Extract the abstract from an Article element."""
        separator = '\n\n'
        abstract = article.find('Abstract')
        abstract = separator.join([a.text for a in abstract.findall('AbstractText')]) if abstract is not None else None
        return abstract
    
    @staticmethod
    def _extract_language(article: ET.Element) -> str | None:
        """Extract the language from an Article element."""
        language = article.find('Language')
        return language.text if language is not None else None
        
    @staticmethod
    def _extract_date(article: ET.Element) -> datetime | None:
        """Extract the publication date from an Article element."""
        date = article.find('ArticleDate')
        if date is None:
            return None
        year = int(date.find('Year').text)
        month = int(date.find('Month').text)
        day = int(date.find('Day').text)
        return datetime(year=year, month=month, day=day)
    
    @staticmethod
    def _extract_publication_types(article: ET.Element) -> List[str]:
        """Extract the publication types from an Article element."""
        return [t.text for t in article.find('PublicationTypeList').findall('PublicationType')]

    def _parse_pubmed_articles(self, batches: List[str]) -> List[Document]:
        """Parse batches of ET.Elements into a single list of Document objects"""
        data = []
        for ib, text in enumerate(batches):
            pubmed_articles = ET.fromstring(text).findall('PubmedArticle')
            logger.debug(f'found #articles={len(pubmed_articles)} in batch {ib}')
            for ie, element in enumerate(pubmed_articles):
                # Extract MedlineCitation and its PMID from PubmedArticle
                citation = self._extract_medline_citation(element=element)
                if citation is None:
                    logger.debug(f'no citation found in PubmedArticle {ie} of batch {ib}')
                    continue
                pmid = self._extract_pmid(element=citation)
                
                # Extract the Article element from the PubmedArticle
                article = self._extract_article(element=citation)
                if article is None:
                    logger.debug(f'no article found in PubmedArticle {ie} of batch {ib}')
                    continue

                # Extract the relevant information from the Article element
                title = self._extract_title(article=article)
                text = self._extract_abstract(article=article)
                if text is None:
                    logger.debug(f'no abstract found in PubmedArticle {ie} of batch {ib}')
                    continue
                language = self._extract_language(article=article)
                publication_date = self._extract_date(article=article)

                # Split long texts into chunks
                texts = self.split_text_into_chunks(text=text)

                # Create the Document object(s)
                for text in texts:
                    document = Document(
                        id_=f'{self._source_url} {title} {text} {language} {publication_date}',
                        text=text,
                        source=self._source,
                        title=title,
                        url=f'{self._source_url}{pmid}/',
                        source_url=self._source_url,
                        source_favicon_url=self._source_favicon_url,
                        language=language,
                        publication_date=publication_date,
                    )
                    data.append(document)
        return data

    

    async def apply(self, term: str, queue: asyncio.Queue) -> None:
        """Query and retrieve all PubmedArticle Documents for the given search term.

        The retrieval is using two main functionalities of the Pubmed API:
        - ESearch: Identify the relevant documents and store them in the entrez history server
        - EFetch: Retrieve the relevant documents from the entrez history server
        """
        logger.debug(f'starting scraping the source={self._source} with term={term}')

        # Search for relevant documents with a given term
        esearch = await self._pubmed_esearch(term=term)

        # Retrieve relevant documents in batches
        params = self._get_entrez_history_params(esearch)
        batches = await self._pubmed_efetch(params=params)

        # Parse documents from batches
        documents = self._parse_pubmed_articles(batches=batches)

        # Add documents to the queue
        for i, doc in enumerate(documents):
            id_ = f'{self._source}_{i}'
            item = QueueItem(id_=id_, doc=doc)
            await queue.put(item)
        
        logger.info(f'found #docs={len(documents)} in source={self._source} for term={term}')


class EMAScraper(Scraper):
    _api_search_url_template = "https://www.ema.europa.eu/en/search?search_api_fulltext={term}&f%5B0%5D=ema_search_entity_is_document%3ADocument"
    
    def get_source_favicon_url(self) -> str:
        return 'https://www.ema.europa.eu/themes/custom/ema_theme/favicon.ico'


def create_tasks(args_: Namespace, queue: asyncio.Queue) -> List[asyncio.Task]:
    """Create the asyncio scraping tasks."""
    sources = args_.source if args_.source is not None else SCRAPING_SOURCES
    term = args_.term

    scrapers = []       # type: List[Scraper]
    if 'pubmed' in sources:
        scrapers.append(PubmedScraper())
    elif 'ema' in sources:
        raise ValueError('EMA source not implemented yet')
    else:
        raise ValueError(f'Unknown source {sources}')
    
    tasks = [asyncio.create_task(scp.apply(term=term, queue=queue)) for scp in scrapers]
    return tasks
