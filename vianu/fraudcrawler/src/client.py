import logging
import asyncio

import pandas as pd

from vianu.fraudcrawler.src.serpapi import SerpApiClient
from vianu.fraudcrawler.src.zyteapi import ZyteAPIClient
from vianu.fraudcrawler.src.processor import Processor

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
        execution_mode: str = "async",
    ):
        """Initializes the Crawler.

        Args:
            serpapi_key: The API key for SERP API.
            zyteapi_key: The API key for Zyte API
            location: The location to use for the search (default: "Switzerland").
            max_retries: Maximum number of retries for API calls (default: 1).
            retry_delay: Delay between retries in seconds (default: 10).
            execution_mode: The default execution mode for fetching product details, either "async" or "sequential" (default: "async").

        """
        self._serpapi_client = SerpApiClient(api_key=serpapi_key, location=location)
        self._zyteapi_client = ZyteAPIClient(
            api_key=zyteapi_key, max_retries=max_retries, retry_delay=retry_delay
        )
        self._processor = Processor(location=location)
        self._execution_mode = execution_mode

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

        # Get product details
        if self._execution_mode == "async":
            queue_in = asyncio.Queue()
            queue_out = asyncio.Queue()

            # Put URLs into the input queue
            for url in urls:
                queue_in.put_nowait(url)  # Non-blocking put

            # Run the async function and get results
            asyncio.run(self._zyteapi_client.async_get_details(queue_in, queue_out))

            # Collect the results
            products = []
            while not queue_out.empty():
                products.append(queue_out.get_nowait())  # Non-blocking get
        elif self._execution_mode == "sequential":
            products = self._zyteapi_client.get_details(urls=urls)
        else:
            NotImplementedError(
                f"Execution mode {self._execution_mode} not implemented yes."
            )

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
