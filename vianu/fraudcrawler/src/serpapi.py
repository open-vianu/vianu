from copy import deepcopy
import logging
import requests
from typing import List

logger = logging.getLogger(__name__)


class SerpApiClient:
    """A client to interact with the SERP API for performing searches."""

    _endpoint = "https://serpapi.com/search"
    _engine = "google"
    _requests_timeout = 10

    def __init__(self, api_key: str, location: str = "Switzerland"):
        """Initializes the SerpApiClient with the given API key.

        Args:
            api_key: The API key for SERP API.
            location: The location to use for the search (default: "Switzerland").
        """
        self._api_key = api_key
        self._location = location
        self._base_config = {
            "engine": self._engine,
            "api_key": api_key,
            "location_requested": self._location,
            "location_used": self._location,
        }

    def search(self, search_term: str, num_results: int = 10) -> List[str]:
        """Performs a search using SERP API and returns the URLs of the results.

        Args:
            search_term: The search term to use for the query.
            num_results: Max number of results to return (default: 10).
        """
        # Setup the parameters
        logger.info(f'Performing SERP API search for search_term "{search_term}".')
        params = deepcopy(self._base_config)
        params["q"] = search_term
        params["num"] = num_results

        # Perform the request
        response = requests.get(
            url=self._endpoint,
            params=params,
            timeout=self._requests_timeout,
        )

        # Handle the response
        status_code = response.status_code
        if status_code == 200:
            data = response.json()
            search_results = data.get("organic_results", [])
            urls = [result.get("link") for result in search_results]
            logger.info(f"Found {len(urls)} URLs from SERP API search.")
            return urls
        else:
            logger.error(f"SERP API request failed with status code {status_code}.")
            return []
