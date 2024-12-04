from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging
from pydantic import BaseModel
import re
import requests
from typing import List
import xml.etree.ElementTree as ET

from .data_model import DOCUMENT_SOURCES, Document, FileHandler
from ..settings import PUBMED_ESEARCH_URL, PUBMED_DB, PUBMED_EFETCH_URL, PUBMED_BATCH_SIZE

MODULE_NAME = 'scraping'

class PubmedEntrezHistoryParams(BaseModel):
    """Class for optimizing Pubmed database for large numbers of documents.
    An example can be found [here](https://www.ncbi.nlm.nih.gov/books/n/helpeutils/chapter3/#chapter3.Application_3_Retrieving_large)
    """
    web: str
    key: str
    count: int


class PubmedClient:

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
        url = f'{PUBMED_ESEARCH_URL}?db={PUBMED_DB}&term={term}&usehistory=y'
        logging.debug(f'search pubmed database with url={url}')
        return requests.get(url=url)

    @staticmethod
    def _pubmed_efetch(params: PubmedEntrezHistoryParams) -> List[requests.Response]:
        logging.debug(f'fetch {params.count} documents in {params.count // PUBMED_BATCH_SIZE + 1} batch(es) of size <= {PUBMED_BATCH_SIZE}')
        batches = []
        for retstart in range(0, int(params.count), PUBMED_BATCH_SIZE):
            url = f'{PUBMED_EFETCH_URL}?db={PUBMED_DB}&WebEnv={params.web}&query_key={params.key}&retstart={retstart}&retmax={PUBMED_BATCH_SIZE}'
            logging.debug(f'fetch documents with url={url}')
            efetch = requests.get(url=url)
            if efetch.status_code == 200:
                batches.append(efetch)
            else:
                logging.error(f'batch failed with code={efetch.status_code}')
        return batches
    
    @staticmethod
    def _extract_article(element: ET.Element) -> ET.Element:
        citation = element.find('MedlineCitation')
        if citation is None:
            logging.warning('no "MedlineCitation" element found')
            return None
        article = citation.find('Article')
        if article is None:
            logging.warning('no "Article" element found')
        return article

    @staticmethod
    def _extract_title(article: ET.Element) -> str | None:
        title = article.find('ArticleTitle')
        return title.text if title is not None else None

    @staticmethod
    def _extract_abstract(article: ET.Element) -> str | None:
        separator = '\n\n'
        abstract = article.find('Abstract')
        abstract = separator.join([a.text for a in abstract.findall('AbstractText')]) if abstract is not None else None
        return abstract
    
    @staticmethod
    def _extract_language(article: ET.Element) -> str | None:
        language = article.find('Language')
        return language.text if language is not None else None
        
    @staticmethod
    def _extract_date(article: ET.Element) -> datetime | None:
        date = article.find('ArticleDate')
        if date is None:
            return None
        year = int(date.find('Year').text)
        month = int(date.find('Month').text)
        day = int(date.find('Day').text)
        return datetime(year=year, month=month, day=day)
    
    @staticmethod
    def _extract_publication_types(article: ET.Element) -> List[str]:
        return [t.text for t in article.find('PublicationTypeList').findall('PublicationType')]

    def _parse_pubmed_articles(self, batches: List[requests.Response], publication_type_filter: List[str] | None = None) -> List[Document]:
        """Parse batches of ET.Elements into a single list of Document objects"""
        documents = []

        for batch in batches:
            pubmed_articles = ET.fromstring(batch.text).findall('PubmedArticle')
            for element in pubmed_articles:
                article = self._extract_article(element=element)
                if article is not None:
                    title = self._extract_title(article=article)
                    abstract = self._extract_abstract(article=article)
                    language = self._extract_language(article=article)
                    publication_date = self._extract_date(article=article)
                    publication_types = self._extract_publication_types(article=article)
                    
                    if publication_type_filter is None or set(publication_type_filter).issubset(set(publication_types)):
                        document = Document(
                            id_=f'{title} {abstract} {language} {publication_date}',
                            source_url=self.source_url,
                            source=self.source,
                            language=language,
                            title=title,
                            abstract=abstract,
                            publication_date=publication_date,
                        )
                        document.add_raw_text(abstract)
                        documents.append(document)
        return documents

    

    def get_pubmed_articles(self, term: str, types: List[str] | None = None) -> List[Document]:
        """Query and retrieve all PubmedArticle Documents for the given search term.

        The retrieval is using two main functionalities of the Pubmed API:
        - ESearch: Identify the relevant documents
        - EFetch: Retrieve the relevant documents
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
    group.add_argument('--source', dest='source', choices=DOCUMENT_SOURCES)
    group.add_argument('--term', dest='term')
    return parser


def apply(args_: Namespace, data: List[Document] | None = None, save_data: bool = True) -> None:
    source = args_.source
    term = args_.term

    if data is None:
        data = FileHandler(args_.data_load).read()

    logging.info(f'crawling source "{source}" for term "{term}"')

    if source == 'pubmed':
        client = PubmedClient()
        articles = client.get_pubmed_articles(term=args_.term)
        logging.info(f'found {len(articles)} pubmed articles')
        data.extend(articles)
    else:
        logging.error(f'Unknown source {source}')
    
    if save_data:
        FileHandler(args_.data_dump).write(data)
