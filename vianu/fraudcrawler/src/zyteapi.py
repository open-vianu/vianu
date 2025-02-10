import asyncio
from copy import deepcopy
import logging
import requests
from requests.auth import HTTPBasicAuth
import time
from tqdm.auto import tqdm
from typing import List
import aiohttp

logger = logging.getLogger(__name__)


class ZyteAPIClient:
    """A client to interact with the Zyte API for fetching product details."""

    _endpoint = "https://api.zyte.com/v1/extract"
    _config = {
        "javascript": False,
        "browserHtml": False,
        "screenshot": False,
        "productOptions": {"extractFrom": "httpResponseBody"},
        "httpResponseBody": True,
        "geolocation": "CH",
        "viewport": {"width": 1280, "height": 1080},
        "actions": [],
    }
    _requests_timeout = 10

    def __init__(
        self,
        api_key: str,
        max_retries: int = 1,
        retry_delay: int = 10,
        async_limit_per_host: int = 5,
    ):
        """Initializes the ZyteApiClient with the given API key and retry configurations.

        Args:
            api_key: The API key for Zyte API.
            max_retries: Maximum number of retries for API calls (default: 1).
            retry_delay: Delay between retries in seconds (default: 10).
            async_limit_per_host: Maximum number of concurrent requests per host for async calls (default: 5).
        """
        self._auth = HTTPBasicAuth(api_key, "")
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._async_limit_per_host = async_limit_per_host

    def get_details(self, urls: List[str], product: bool = True) -> List[dict]:
        """Fetches product details from the given URLs using Zyte API.

        Args:
            urls: A list of URLs to fetch product details from.
            product: Whether to extract product details (default: True).
        """
        logger.info(
            f"fetching product details for {len(urls)} URLs via Zyte API (synchronous)"
        )

        config = deepcopy(self._config)
        config["product"] = product
        products = []
        with tqdm(total=len(urls)) as pbar:
            for url in urls:
                attempts = 0
                while attempts < self._max_retries:
                    try:
                        logger.debug(
                            f"fetch product details for URL {url} (Attempt {attempts + 1})"
                        )

                        response = requests.post(
                            self._endpoint,
                            auth=self._auth,
                            json={
                                "url": url,
                                **config,
                            },
                            timeout=self._requests_timeout,
                        )
                        if response.status_code == 200:
                            product_data = response.json()
                            product_data["url"] = url  # Ensure the URL is included
                            products.append(product_data)
                            logger.debug(
                                f"successfully fetched product details for URL {url}"
                            )
                            break
                        else:
                            logger.error(
                                f"Zyte API request failed for URL {url} with status code {response.status_code} "
                                f"and response: {response.text}"
                            )
                            attempts += 1
                            if attempts < self._max_retries:
                                logger.warning(
                                    f"retrying in {self._retry_delay} seconds..."
                                )
                                time.sleep(self._retry_delay)
                    except Exception as e:
                        logger.error(
                            f"exception occurred while fetching product details for URL {url}: {e}"
                        )
                        attempts += 1
                        if attempts < self._max_retries:
                            logger.warning(
                                f"retrying in {self._retry_delay} seconds..."
                            )
                            time.sleep(self._retry_delay)
                else:
                    logger.error(f"all attempts failed for URL: {url}")
                pbar.update(1)

        logger.info(f"fetched product details for {len(products)} URLs")
        return products

    async def async_get_details(
        self, queue_in: asyncio.Queue, queue_out: asyncio.Queue
    ) -> None:
        """Fetches product details from the URLs in the input queue using Zyte API and puts the results into the output queue.

        Args:
            queue_in: the input queue containing URLs to fetch product details from
            queue_out: the output queue to put the product details into
        """

        # Drain the input queue into a list of URLs.
        urls = []
        while not queue_in.empty():
            url = await queue_in.get()
            urls.append(url)

        # Use the same configuration as the synchronous version.
        config = deepcopy(self._config)
        config["product"] = True

        connector = aiohttp.TCPConnector(limit_per_host=self._async_limit_per_host)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._aget_details_for_url(session, url, config) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error fetching details: {result}")
                else:
                    await queue_out.put(result)

    async def _aget_details_for_url(
        self, session: aiohttp.ClientSession, url: str, config: dict
    ) -> dict:
        """Helper coroutine to fetch product details for a single URL using aiohttp.

        Args:
            session: The aiohttp ClientSession to use for the request.
            url: The URL to fetch product details from.
            config: The configuration dictionary for the API request.

        Returns:
            A dictionary with the product details.
        """
        attempts = 0
        while attempts < self._max_retries:
            try:
                logger.debug(
                    f"fetch product details for URL {url} (Attempt {attempts + 1})"
                )
                async with session.post(
                    self._endpoint,
                    json={"url": url, **config},
                    auth=aiohttp.BasicAuth(self._auth.username, self._auth.password),
                    timeout=self._requests_timeout,
                ) as response:
                    if response.status == 200:
                        product_data = await response.json()
                        product_data["url"] = url  # Ensure the URL is included
                        logger.debug(
                            f"successfully fetched product details for URL {url}"
                        )
                        return product_data
                    else:
                        text = await response.text()
                        logger.error(
                            f"Zyte API request failed for URL {url} with status code {response.status} "
                            f"and response: {text}"
                        )
                        attempts += 1
                        if attempts < self._max_retries:
                            logger.warning(
                                f"retrying in {self._retry_delay} seconds..."
                            )
                            await asyncio.sleep(self._retry_delay)
            except Exception as e:
                logger.error(
                    f"exception occurred while fetching product details for URL {url}: {e}"
                )
                attempts += 1
                if attempts < self._max_retries:
                    logger.warning(f"retrying in {self._retry_delay} seconds...")
                    await asyncio.sleep(self._retry_delay)
        else:
            logger.error(f"all attempts failed for URL: {url}")
            return {"url": url, "error": "failed after max retries"}
