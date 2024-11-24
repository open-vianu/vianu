import requests
import logging

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
