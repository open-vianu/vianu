from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging
import numpy as np
from pydantic import BaseModel
import re
import requests
from typing import List
import xml.etree.ElementTree as ET

from .data_model import Document, FileHandler
from ..settings import PUBMED_ESEARCH_URL, PUBMED_DB, PUBMED_EFETCH_URL, PUBMED_BATCH_SIZE, DEFAULT_MAX_TOKENS

logger = logging.getLogger(__name__)

MODULE_NAME = 'scraping'
_DEFAULT_MAX_SIZE = round(DEFAULT_MAX_TOKENS / 0.8)


class Scraper(ABC):

    _word_separator = ' '

    @abstractmethod
    def apply(self, term: str) -> List[Document]:
        pass


    def get_text_chunks(self, text: str, max_size: int = _DEFAULT_MAX_SIZE) -> List[str]:
        """Split a text into chunks of a given max size."""
        words = text.split(self._word_separator)
        N = len(words)
        s = min(max_size, N)
        n = N // s
        bnd = [round(i) for i in np.linspace(0, 1, n+1) * N]

        texts = []
        for start, stop in zip(bnd[:-1], bnd[1:]):
            texts.append(self._word_separator.join(words[start:stop]))
        return texts


class PubmedEntrezHistoryParams(BaseModel):
    """Class for optimizing Pubmed database for large numbers of documents.
    An example can be found [here](https://www.ncbi.nlm.nih.gov/books/n/helpeutils/chapter3/#chapter3.Application_3_Retrieving_large)
    """
    web: str
    key: str
    count: int


class PubmedScraper(Scraper):

    source = 'Pubmed'
    source_url = 'https://pubmed.ncbi.nlm.nih.gov/'

    @staticmethod
    def _get_entrez_history_params(text: str) -> PubmedEntrezHistoryParams:
        """Retrieving the entrez history parameters for optimized search when requesting large numbers of documents.
        An example can be found [here](https://www.ncbi.nlm.nih.gov/books/n/helpeutils/chapter3/#chapter3.Application_3_Retrieving_large)
        """
        web = re.search(r'<WebEnv>(\S+)<\/WebEnv>', text).group(1)
        key = re.search(r'<QueryKey>(\d+)<\/QueryKey>', text).group(1)
        count = int(re.search(r'<Count>(\d+)<\/Count>', text).group(1))
        return PubmedEntrezHistoryParams(web=web, key=key, count=count)
    
    @staticmethod
    def _pubmed_esearch(term: str) -> requests.Response:
        """Search the Pubmed database with a given term and POST the results to entrez history server."""
        url = f'{PUBMED_ESEARCH_URL}?db={PUBMED_DB}&term={term}&usehistory=y'
        logger.debug(f'search pubmed database with url={url}')
        return requests.get(url=url)

    @staticmethod
    def _pubmed_efetch(params: PubmedEntrezHistoryParams) -> List[requests.Response]:
        """Retrieve the relevant documents from the entrez history server."""
        logger.debug(f'fetch {params.count} documents in {params.count // PUBMED_BATCH_SIZE + 1} batch(es) of size <= {PUBMED_BATCH_SIZE}')
        batches = []
        for retstart in range(0, int(params.count), PUBMED_BATCH_SIZE):
            url = f'{PUBMED_EFETCH_URL}?db={PUBMED_DB}&WebEnv={params.web}&query_key={params.key}&retstart={retstart}&retmax={PUBMED_BATCH_SIZE}'
            logger.debug(f'fetch documents with url={url}')
            efetch = requests.get(url=url)
            if efetch.status_code == 200:
                batches.append(efetch)
            else:
                logger.error(f'batch failed with code={efetch.status_code}')
        return batches
    
    @staticmethod
    def _extract_article(element: ET.Element) -> ET.Element | None:
        """Extract the article element from a PubmedArticle element."""

        # Find and extract the MedlineCitation element
        citation = element.find('MedlineCitation')
        if citation is None:
            logger.warning('no "MedlineCitation" element found')
            return None
        
        # Find and extract the Article element
        article = citation.find('Article')
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

    def _parse_pubmed_articles(self, batches: List[requests.Response], publication_type_filter: List[str] | None = None) -> List[Document]:
        """Parse batches of ET.Elements into a single list of Document objects"""
        documents = []
        for batch in batches:
            pubmed_articles = ET.fromstring(batch.text).findall('PubmedArticle')
            for element in pubmed_articles:
                article = self._extract_article(element=element)
                if article is not None:
                    # Extract the relevant information from the Article element
                    title = self._extract_title(article=article)
                    text = self._extract_abstract(article=article)
                    if text is None:
                        continue
                    language = self._extract_language(article=article)
                    publication_date = self._extract_date(article=article)

                    # Split long texts into chunks
                    texts = self.get_text_chunks(text=text)

                    # Create the Document object(s)
                    for text in texts:
                        document = Document(
                            id_=f'{self.source_url} {title} {text} {language} {publication_date}',
                            text=text,
                            source=self.source,
                            title=title,
                            source_url=self.source_url,
                            language=language,
                            publication_date=publication_date,
                        )
                        documents.append(document)
        return documents

    

    def apply(self, term: str) -> List[Document]:
        """Query and retrieve all PubmedArticle Documents for the given search term.

        The retrieval is using two main functionalities of the Pubmed API:
        - ESearch: Identify the relevant documents and store them in the entrez history server
        - EFetch: Retrieve the relevant documents from the entrez history server
        """

        # Search for relevant documents with a given term
        esearch = self._pubmed_esearch(term=term)

        # Retrieve relevant documents in batches
        params = self._get_entrez_history_params(esearch.text)
        batches = self._pubmed_efetch(params=params)

        # Parse documents from batches
        documents = self._parse_pubmed_articles(batches=batches)
        return documents


def cli_args() -> None:
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    group.add_argument('--source', dest='source', choices=['pubmed', 'faers'])
    group.add_argument('--term', dest='term')
    return parser


def apply(args_: Namespace, save_data: bool = True) -> List[Document]:
    source = args_.source
    term = args_.term
    data = []

    logger.info(f'crawling source "{source}" for term "{term}"')
    if source == 'pubmed':
        scraper = PubmedScraper()
        documents = scraper.apply(term=args_.term)
        logger.info(f'found {len(documents)} pubmed articles')
        data.extend(documents)
    elif source == 'faers':
        logger.error('FAERS source not implemented yet')
    else:
        logger.error(f'Unknown source {source}')
    
    if save_data:
        FileHandler(args_.data_dump).write(data)
    
    return data
