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
        retry_delay: int = 2,
    ):
        """Initializes the Crawler.

        Args:
            serpapi_key: The API key for SERP API.
            zyteapi_key: The API key for Zyte API
            location: The location to use for the search (default: "Switzerland").
            max_retries: Maximum number of retries for API calls (default: 1).
            retry_delay: Delay between retries in seconds (default: 2).

        """
        self._serpapi_client = SerpApiClient(api_key=serpapi_key, location=location)
        self._zyteapi_client = ZyteAPIClient(
            api_key=zyteapi_key, max_retries=max_retries, retry_delay=retry_delay
        )
        self._processor = Processor(location=location)

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

    async def collect(self, queue_in: asyncio.Queue) -> None:
        while True:
            item = await queue_in.get()
            if item is None:
                queue_in.task_done()
                break
            logging.info(f"Collected item: {str(item)[:100]}...")
            queue_in.task_done()

    async def async_run(
        self,
        search_term: str,
        num_results=10,
        n_zyte_workers: int = 5,
        n_processor_workers: int = 5,
    ) -> None:
        """Runs the pipeline steps: search, get product details, processes them, and returns a DataFrame.

        Args:
            search_term: The search term for the query.
            num_results: Max number of search results (default: 10).
            n_zyte_workers: Number of async workers for zyte (default: 5).
            n_processor_workers: Number of async workers for the processor (default: 5).
        """
        # Perform search
        urls = self._serpapi_client.search(
            search_term=search_term,
            num_results=num_results,
        )
        if not urls:
            logger.warning("No URLs found from SERP API.")
            return pd.DataFrame()

        queue_urls = asyncio.Queue()
        queue_products = asyncio.Queue()
        queue_results = asyncio.Queue()

        # Put URLs into the input queue
        for url in urls:
            await queue_urls.put(url)

        # Put n_zyte_workers as stopping creteria (sentinel)
        for _ in range(n_zyte_workers):
            await queue_urls.put(None)

        zyte_workers = [
            asyncio.create_task(
                self._zyteapi_client.async_get_details(
                    queue_in=queue_urls, queue_out=queue_products
                )
            )
            for _ in range(n_zyte_workers)
        ]

        processor_workers = [
            asyncio.create_task(
                self._processor.async_process(
                    queue_in=queue_products, queue_out=queue_results
                )
            )
            for _ in range(n_processor_workers)
        ]

        collector_worker = asyncio.create_task(self.collect(queue_results))

        # Await that all zyte-workers are finished
        await asyncio.gather(*zyte_workers)
        for worker in zyte_workers:
            worker.cancel()

        # After all zyte workers are finished, put n_processor_workers as stopping cretieria (sentinel)
        for _ in range(n_processor_workers):
            await queue_products.put(None)

        # Log all workers are done
        await asyncio.gather(*processor_workers)
        for worker in processor_workers:
            worker.cancel()

        # Put 1 sentinel for results
        await queue_results.put(None)

        # After the sine collector worker is done, stop him
        await collector_worker
        collector_worker.cancel()

        logger.info("All workers have finished and stopped")
