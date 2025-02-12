import pandas as pd
import logging
import os

import pandas as pd

from vianu.fraudcrawler.src.serpapi import SerpApiClient
from vianu.fraudcrawler.src.zyteapi import ZyteAPIClient
from vianu.fraudcrawler.src.processor import Processor
from vianu.fraudcrawler.src.enrichment import KeywordEnricher

logger = logging.getLogger(__name__)


class FraudCrawlerClient:
    """The main client that orchestrates the search, data fetching, and processing."""

    def __init__(
            self,
            serpapi_key: str,
            zyteapi_key: str,
            location: str = "Switzerland",
            max_retries: int = 3,
            retry_delay: int = 10,
        ):
        """Initializes the Crawler.

        Args:
            serpapi_key: The API key for SERP API.
            zyteapi_key: The API key for Zyte API
            location: The location to use for the search (default: "Switzerland").
            max_retries: Maximum number of retries for API calls (default: 1).
            retry_delay: Delay between retries in seconds (default: 10).
        """
        self._serpapi_client = SerpApiClient(api_key=serpapi_key, location=location)
        self._zyteapi_client = ZyteAPIClient(api_key=zyteapi_key, max_retries=max_retries, retry_delay=retry_delay)
        self._processor = Processor(location=location)

    # allow_enrichment: bool
    def run(self, search_term: str, num_results=10) -> pd.DataFrame:
        """Runs the pipeline steps: search, get product details, processes them, and returns a DataFrame.

        Args:
            search_term: The search term for the query.
            num_results: Max number of search results (default: 10).
        """
        # Perform search
        urls = self._serpapi_client.search(
            search_term=search_term,
            num_results=num_results,
        )
        if not urls:
            logger.warning("No URLs found from SERP API.")
            return pd.DataFrame()
        Enricher = KeywordEnricher(self.serpapi_token, self.zyte_api_key)

        # Make enrichment
        if allow_enrichment:
            added_enriched_words = 2
            added_urls_per_kw = 3
            enhanced_df = Enricher.apply(query,
                                         added_enriched_words,
                                         location,
                                         'German',
                                         added_urls_per_kw)
            urls = urls + enhanced_df["url"].tolist()

        # Get product details
        products = self._zyteapi_client.get_details(urls=urls)
        if not products:
            logger.warning("No product details fetched from Zyte API.")
            return pd.DataFrame()

        # Process products
        processed = self._processor.process(products=products)
        if not processed:
            logger.warning("No products left after processing.")
            return pd.DataFrame()

        # Flatten the product data
        df = pd.json_normalize(processed)

        # Log and return the DataFrame
        logger.info("Search completed. Returning flattened DataFrame.")
        return df
