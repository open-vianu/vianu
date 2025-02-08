import asyncio
import logging
from typing import List

import pandas as pd

from vianu.fraudcrawler.src.serpapi import SerpApiClient
from vianu.fraudcrawler.src.zyteapi import ZyteAPIClient

logger = logging.getLogger(__name__)


class Crawler:
    """The main client that orchestrates the search (SerpAPI) and data fetching (ZyteAPI)."""

    _location_mapping = {
        "Switzerland": "ch",
        "Chile": "cl",
        "Austria": "at",
    }

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
            serpapi_key: the API key for SERP API
            zyteapi_key: the API key for Zyte API
            location: the location to use for the search (default: "Switzerland")
            max_retries: maximum number of retries for API calls (default: 1)
            retry_delay: delay between retries in seconds (default: 10)
        """
        self._serpapi_client = SerpApiClient(api_key=serpapi_key, location=location)
        self._zyteapi_client = ZyteAPIClient(api_key=zyteapi_key, max_retries=max_retries, retry_delay=retry_delay)
        country_code = self._location_mapping.get(location)
        if country_code is None:
            logger.warning(
                f'location="{location}" not found in self._location_mapping (defaulting to "ch")'
            )
            country_code = "ch"
        self._country_code = country_code.lower()

    def _keep_product(self, product: dict) -> bool:
        """Determines whether to keep the product based on the filtering criteria.

        Args:
            product: a product data dictionary
        """
        url = product.get("url", "")
        return (
            f".{self._country_code}/" in url.lower()
            or url.lower().endswith(f".{self._country_code}")
            or ".com" in url.lower()
        )

    def _filter_products(self, products: List[dict]) -> List[dict]:
        """Filters the products based on the country code.

        Args:
            products: a list of product data dictionaries
        """
        logger.info(
            f'filtering {len(products)} products by country_code="{self._country_code.upper()}"'
        )
        filtered = [prod for prod in products if self._keep_product(prod)]
        logger.info(
            f"filtered down to {len(filtered)} products after applying country code filter"
        )
        return filtered

    def apply(self, search_term: str, num_results=10) -> pd.DataFrame:
        """Performs the search, gets product details, processes them, and returns a DataFrame.

        Args:
            search_term: the search term for the query
            num_results: number of search results to process (default: 10)
        """
        # Perform search
        urls = self._serpapi_client.search(
            search_term=search_term,
            num_results=num_results,
        )
        if not urls:
            logger.warning("no URLs found from SERP API")
            return pd.DataFrame()

        # Get product details
        products = self._zyteapi_client.get_details(urls=urls)
        if not products:
            logger.warning("no product details fetched from Zyte API")
            return pd.DataFrame()

        # Process products
        filtered = self._filter_products(products=products)
        if not filtered:
            logger.warning("no products left after filtering")
            return pd.DataFrame()

        # Flatten the product data
        df = pd.json_normalize(filtered)

        # Log and return the DataFrame
        logger.info("crawling completed successfully")
        return df

    async def aapply(self, queue_out: asyncio.Queue, search_term: str, num_results=10) -> None:
        """Performs async search and puts the results into the output queue.
        
        Args:
            queue_out: the output queue to put the results
            search_term: the search term for the query
            num_results: number of search results to process (default: 10)
        """
        # queue_mid = asyncio.Queue()
        raise NotImplementedError("Crawler.aapply method is not implemented yet")
    