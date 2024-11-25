from argparse import ArgumentParser
from datetime import datetime
import logging
import re
import requests
from typing import List
import xml.etree.ElementTree as ET

from .data_model import DOCUMENT_SOURCES, Document
from ..settings import PUBMED_ESEARCH_URL, PUBMED_DB, PUBMED_EFETCH_URL, PUBMET_BATCH_SIZE

MODULE_NAME = 'scraping'


class PubmedClient:

    @staticmethod
    def _get_history_parameters(text: str) -> dict:
        web = re.search(r'<WebEnv>(\S+)<\/WebEnv>', text)
        web = web.group(1) if web else None
        key = re.search(r'<QueryKey>(\d+)<\/QueryKey>', text)
        key = key.group(1) if key else None
        count = re.search(r'<Count>(\d+)<\/Count>', text)
        count = count.group(1) if count else None

        return {'web': web, 'key': key, 'count': count}
    

    def get_documents(self, term: str) -> List[Document]:
        url = f'{PUBMED_ESEARCH_URL}?db={PUBMED_DB}&term={term}&usehistory=y'
        esearch = requests.get(url=url)
        params = self._get_history_parameters(esearch.text)

        content = []
        for retstart in range(0, int(params['count']), PUBMET_BATCH_SIZE):
            url = f'{PUBMED_EFETCH_URL}?db={PUBMED_DB}&WebEnv={params["web"]}'\
                  f'&query_key={params["key"]}&retstart={retstart}&retmax={PUBMET_BATCH_SIZE}'
            efetch = requests.get(url=url)
            content.append(efetch.text)
        
        documents = []
        for c in content:
            for article in ET.fromstring(c).findall('PubmedArticle'):
                citation = article.find('MedlineCitation')
                article = citation.find('Article')
                title = article.find('ArticleTitle').text
                abstract = article.find('Abstract')
                abstract = '\n'.join([a.text for a in abstract.findall('AbstractText')]) if abstract else None
                language = article.find('Language').text

                date = article.find('ArticleDate')
                if date:
                    year = int(date.find('Year').text)
                    month = int(date.find('Month').text)
                    day = int(date.find('Day').text)
                    publication_date = datetime(year=year, month=month, day=day)
                else:
                    publication_date = None

                publication_types = [t.text for t in article.find('PublicationTypeList').findall('PublicationType')]

                documents.append(Document(
                    id_=f'{title} {abstract}',
                    title=title,
                    abstract=abstract,
                    publication_date=publication_date,
                ))
        return documents


def add_parser(parent: ArgumentParser):
    subparser = parent.add_parser(MODULE_NAME)
    subparser.add_argument('--source', dest='source', choices=DOCUMENT_SOURCES)
    subparser.add_argument('--term', dest='term')


def apply(args_):
    source = args_.source
    term = args_.term
    logging.info(f'crawling source "{source}" for term "{term}"')
    
    if source == 'pubmed':
        client = PubmedClient()
        documents = client.get_documents(term=args_.term)
        logging.info(f'found {len(documents)} articles')
    else:
        logging.error(f'Unknown source {source}')
