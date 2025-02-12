import requests
import logging
import hashlib
import time
import re
import json
from typing import Any, Dict, Callable, List
from serpapi.google_search import GoogleSearch
from typing import Optional
from requests import Response
from urllib.parse import quote_plus


logger = logging.getLogger("fraudcrawler_logger")


class SerpApiClient:
    """
    A client to interact with the SERP API for performing search queries.
    """

    def __init__(self, serpapi_token, location):
        """
        Initializes the SerpApiClient with the given API token.

        Args:
            serpapi_token (str): The API token for SERP API.
        """
        self.serpapi_token = serpapi_token
        self.location = location

    def search(self, query, num_results=10):
        """
        Performs a search using SERP API and returns the URLs of the results.

        Args:
            query (str): The search query.
            num_results (int): Number of results to return.

        Returns:
            list: A list of URLs from the search results.
        """
        logger.info(f"Performing SERP API search for query: {query}")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_token,
            "num": num_results,
            "location_requested": self.location,
            "location_used": self.location,
        }

        response = requests.get("https://serpapi.com/search", params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            search_results = data.get("organic_results", [])
            urls = [result.get("link") for result in search_results]
            logger.info(f"Found {len(urls)} URLs from SERP API.")
            return urls
        else:
            logger.error(
                f"SERP API request failed with status code {response.status_code}"
            )
            return []

    def _generate_hash(self, data: Any) -> str:
        data_str = str(data)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


    def _mask_token_in_string(self, string_to_mask: str, token: str) -> str:
        return re.sub(re.escape(token), f"{re.escape(token[:5])}*****", string_to_mask)

    def convert_request_to_string(self, req: requests.models.PreparedRequest, token_to_mask: Optional[str] = None) -> str:
        result = f"method: {req.method}, url: {req.url}"
        if req.body:
            result += ", body: " + req.body
        if not token_to_mask:
            return result
        return self._mask_token_in_string(result, quote_plus(token_to_mask))

    def convert_response_to_string(self, response: Response, token_to_mask: Optional[str] = None) -> str:
        try:
            # Attempt to get json formatted data from response and turn it to CloudWatch-friendly format
            result = json.dumps(response.json())
        except json.decoder.JSONDecodeError:
            result = response.text

        if not token_to_mask:
            return result
        return self._mask_token_in_string(result, token_to_mask)

    @staticmethod
    def _check_limit(urls: List[str], query: str, limit: int = 200) -> List[str]:
        """
        Checks if the number of URLs exceeds the limit, and trims the list if necessary.

        Args:
            urls (List[str]): The list of URLs.
            query (str): The search query.
            limit (int): hight of limit

        Returns:
            List[str]: The potentially trimmed list of URLs.
        """
        if len(urls) > limit:
            urls = urls[:limit]
            logger.warning(f"Reached limit for keyword: {query}")
        return urls

    @staticmethod
    def get_organic_results(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts the organic search results from the API response.

        Args:
            results (Dict[str, Any]): The JSON response from the API.

        Returns:
            List[Dict[str, Any]]: A list of organic search results.
        """
        return results.get("organic_results") or []

    def call_serpapi(
                self,
                params: Dict[str, Any],
                log_name: str,
                force_refresh: bool = False,
                callback: Callable[int, None] | None = None,
        ) -> Dict[str, Any]:
        """
            Calls the SerpAPI and returns the response, with optional caching.

            Args:
                params (Dict[str, Any]): Parameters for the API call.
                log_name (str): The name used for logging.
                force_refresh (bool): Whether to bypass the cache and force a new API call (default is False).

            Returns:
                Dict[str, Any]: The JSON response from the SerpAPI.

            Raises:
                Exception: If all API call attempts fail.
        """
        data_hash = self._generate_hash(str(params))

        attempts = 0
        max_retries = 5
        retry_delay = 5
        while attempts < max_retries:
            try:
                search = GoogleSearch(params)

                response = search.get_response()

                logger.debug(
                    f'{log_name}: req: {self.convert_request_to_string(response.request, params.get("api_key"))}'
                )
                logger.debug(
                    f"{log_name}: response: \n"
                    + self.convert_response_to_string(response, params.get("api_key"))
                )
                response.raise_for_status()
                if callback is not None:
                    callback(1)
                return response.json()
            except Exception as e:
                logger.warning(
                    f"API call failed with error: {e}. Retrying in {retry_delay} seconds..."
                )
                attempts += 1
                time.sleep(retry_delay)
        raise Exception("All API call attempts to SerpAPI failed.")