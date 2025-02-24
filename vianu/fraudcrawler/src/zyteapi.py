import logging
import requests
from requests.auth import HTTPBasicAuth
import time
from tqdm.auto import tqdm
from typing import List

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
        "product": True,
    }
    _requests_timeout = 10

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """Initializes the ZyteApiClient with the given API key and retry configurations.

        Args:
            api_key: The API key for Zyte API.
            max_retries: Maximum number of retries for API calls (default: 3).
            retry_delay: Delay between retries in seconds (default: 5).
        """
        self._basic_auth = HTTPBasicAuth(api_key, "")
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def get_details(self, urls: List[str], product: bool = True) -> List[dict]:
        """Fetches product details from the given URLs using Zyte API.

        Args:
            urls: A list of URLs to fetch product details from.
            product: Whether to extract product details (default: True).
        """
        logger.info(
            f"fetching product details for {len(urls)} URLs via Zyte API (synchronous)"
        )

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
                            auth=self._basic_auth,
                            json={
                                "url": url,
                                **self._config,
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
