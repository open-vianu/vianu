import requests
import logging
import time
from tqdm.auto import tqdm
from requests.auth import HTTPBasicAuth

logger = logging.getLogger("fraudcrawler_logger")


class ZyteApiClient:
    """
    A client to interact with the Zyte API for fetching product details.
    """

    def __init__(self, zyte_api_key, max_retries=1, retry_delay=10):
        """
        Initializes the ZyteApiClient with the given API key and retry configurations.

        Args:
            zyte_api_key (str): The API key for Zyte API.
            max_retries (int): Maximum number of retries for API calls.
            retry_delay (int): Delay between retries in seconds.
        """
        self.endpoint = "https://api.zyte.com/v1/extract"
        self.auth = HTTPBasicAuth(zyte_api_key, "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_product_details(self, urls):
        """
        Fetches product details from the given URLs using Zyte API.

        Args:
            urls (list): A list of URLs to fetch product details from.

        Returns:
            list: A list of dictionaries containing product details.
        """
        logger.info(f"Fetching product details for {len(urls)} URLs via Zyte API.")
        products = []
        config = {
            "javascript": False,
            "browserHtml": False,
            "screenshot": False,
            "product": True,
            "productOptions": {"extractFrom": "httpResponseBody"},
            "httpResponseBody": True,
            "geolocation": "CH",
            "viewport": {"width": 1280, "height": 1080},
            "actions": [],
        }

        with tqdm(total=len(urls)) as pbar:
            for url in urls:
                attempts = 0
                while attempts < self.max_retries:
                    try:
                        logger.debug(
                            f"Attempting to fetch product details for URL: {url} (Attempt {attempts + 1})"
                        )
                        response = requests.post(
                            self.endpoint,
                            auth=self.auth,
                            json={
                                "url": url,
                                **config,
                            },
                            timeout=10,
                        )

                        if response.status_code == 200:
                            product_data = response.json()
                            product_data["url"] = url  # Ensure the URL is included
                            products.append(product_data)
                            logger.debug(
                                f"Successfully fetched product details for URL: {url}"
                            )
                            break  # Exit the retry loop on success
                        else:
                            logger.error(
                                f"Zyte API request failed for URL {url} with status code {response.status_code} "
                                f"and response: {response.text}"
                            )
                            attempts += 1
                            if attempts < self.max_retries:
                                logger.warning(
                                    f"Retrying in {self.retry_delay} seconds..."
                                )
                                time.sleep(self.retry_delay)
                    except Exception as e:
                        logger.error(
                            f"Exception occurred while fetching product details for URL {url}: {e}"
                        )
                        attempts += 1
                        if attempts < self.max_retries:
                            logger.warning(f"Retrying in {self.retry_delay} seconds...")
                            time.sleep(self.retry_delay)
                else:
                    logger.error(f"All attempts failed for URL: {url}")
                pbar.update(1)

        logger.info(f"Fetched product details for {len(products)} URLs.")
        return products
