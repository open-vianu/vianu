from vianu.fraudcrawler.src.serpapi import SerpApiClient
from typing import Any, Dict, Callable, List
import pandas as pd
from json import dumps
from json import loads
from http.client import HTTPSConnection
from base64 import b64encode
from vianu.fraudcrawler.src import enrichment_utils
import os
import logging
import hashlib


logger = logging.getLogger("fraudcrawler_logger")


class KeywordEnricher:
    def __init__(self, serpapi_key=None, location="Switzerland") -> None:
        """
        Initializes the KeywordEnricher.
        """
        super().__init__()
        self.serpapi_token = serpapi_key
        self.location = location

    def apply(
        self,
        keyword: str,
        number_of_keywords: int,
        language: str,
        added_url_per_kw: int = 10,
        check_limit: int = 200,
    ):
        """
        Makes the API call to SerpAPI to enrich the keyword and retrieve search results.

        The process is decomposed as follows:
        1. From root keyword, call DataforSeo API to get suggested and related keywords with maximum search
        volumes for different locations and languages.
        2. Deduplicate keywords and aggregate search volume from different locations/languages.
        3. Call SerpAPI for selected keywords and get the corresponding URLs (top 20 only).
        4. Deduplicate URLs and estimate total traffic per URL.
        5. Return the top 200 URLs with the highest traffic.

        Args:
            keyword (str): The search keyword to enrich.
            serpapi (SerpAPI): The SerpAPI client instance to fetch results.
            number_of_keywords (int): The number of keywords to retrieve.
            location (str): The location for keyword enrichment.
            language (str): The language to search in.

        Returns:
            PipelineResult: Contains metadata and enriched search results.
        """

        serp_client = SerpApiClient(self.serpapi_token, self.location)

        # Initializes a SUGGESTED_KW and RELATED_KW from the DataforSEO engine.
        # NOTICE:
        #     suggested_kw - is a collection of search queries that INCLUDE the seed keyword.
        #     related_kw - keywords that appear in the "searchers related to" section of the engine.
        suggested_kw: List[Dict[str, Any]] = []
        related_kw: List[Dict[str, Any]] = []

        # Initializes DataforSEO API
        dataforseo_api = DataforSeoAPI()

        suggested_kw += dataforseo_api.get_keyword_suggestions(
            keyword, self.location, language, number_of_keywords
        )

        related_kw += dataforseo_api.get_related_keywords(
            keyword, self.location, language, number_of_keywords
        )

        enriched_kw = suggested_kw + related_kw
        logger.info(
            f"ENRICHMENT: For query {keyword}, total number of additional keywords -> {len(enriched_kw)}"
        )

        filtered_kw: List[str] = []

        # NOW we filtered and deduplicate
        for kw in enriched_kw:
            filtered_kw.append(enrichment_utils.filter_keywords(kw["keywordEnriched"]))

        agg_kw = enrichment_utils.aggregate_keywords(enriched_kw).to_dict(
            orient="records"
        )

        urls: List[Dict[str, Any]] = []
        for kw in agg_kw:
            response = self.retrieve_response(
                keyword=kw["keywordEnriched"],
                client=serp_client,
                added_url_per_kw=added_url_per_kw,
                offer_root=kw["offerRoot"],
            )
            items = serp_client.get_organic_results(response)

            kw_urls = [item.get("link") for item in items]
            results = serp_client._check_limit(kw_urls, kw, check_limit)
            urls += enrichment_utils.estimate_volume_per_url(
                results,
                kw["keywordVolume"],
                kw["keywordEnriched"],
                kw["keywordLocation"],
                kw["keywordLanguage"],
                kw["offerRoot"],
            )

        # AGGREGATES RESULTS (group-by by URLs)
        enriched_results = enrichment_utils.aggregate_urls(urls)

        return pd.DataFrame(enriched_results)

    def retrieve_response(
        self,
        keyword: str,
        client: SerpApiClient,
        added_url_per_kw: int,
        offer_root: str = "DEFAULT",
        callback: Callable[int, None] | None = None,
        custom_params: Dict[str, Any] = {},
    ) -> List:
        params = {
            "q": keyword,
            "start": 0,
            "api_key": self.serpapi_token,
            "num": added_url_per_kw,
            **(custom_params),
        }
        logger.info(
            f"ENRICHMENT: Extracting URLs from SerpAPI for '{keyword}' from '{offer_root}'"
        )
        return client.call_serpapi(params, log_name="google_regular", callback=callback)


class DataforSeoAPI:
    def __init__(
        self,
        cache_name: str = "dataforseoapi",
        max_retries: int = 3,
        retry_delay: int = 2,
        cache_duration: int = 24 * 60 * 60,
    ):
        """
        Initializes the SerpAPI class.

        Args:
            cache_name (str): The name of the cache (default is "serpapi").
            max_retries (int): The maximum number of retries for API calls (default is 3).
            retry_delay (int): The delay in seconds between retry attempts (default is 2).
            cache_duration (int): The delay in seconds between a cache entry is considered obsolete.

        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_name = cache_name
        self.cache_duration = cache_duration

    @staticmethod
    def _generate_hash(data: Any) -> str:
        data_str = str(data)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def request(self, path, method, data=None):
        """Make a request to the DataforSEO API

        Args:
            path (str): path to the API endpoint
            method (str): HTTP method
            data (str): data to send with the request

        Returns:
            dict with the response from the API"""

        connection = HTTPSConnection("api.dataforseo.com")
        try:
            base64_bytes = b64encode(
                (
                    "%s:%s"
                    % (
                        os.getenv("DATAFORSEO_USERNAME", "YOUR_DATAFORSEO_USERNAME"),
                        # self.context.settings.data_for_seo.username,
                        os.getenv("DATAFORSEO_PASSWORD", "YOUR_DATAFORSEO_PASSWORD"),
                        # self.context.settings.data_for_seo.password,
                    )
                ).encode("ascii")
            ).decode("ascii")
            headers = {
                "Authorization": "Basic %s" % base64_bytes,
                "Content-Encoding": "gzip",
            }
            connection.request(method, path, headers=headers, body=data)
            response = connection.getresponse()
            if response.status >= 400:
                raise Exception(
                    f"Failed to call dataforseo: {response.status} {response.reason}"
                )
            return loads(response.read().decode())
        finally:
            connection.close()

    def get(self, path):
        return self.request(path, "GET")

    def post(self, path, data):
        if isinstance(data, str):
            data_str = data
        else:
            data_str = dumps(data)
        return self.request(path, "POST", data_str)

    def get_keyword_suggestions(self, keyword, location_name, language_name, limit=100):
        """Get keyword suggestions for a given keyword, location and language

        Args:
            keyword (str): keyword to get suggestions for
            location_name (str): location name
            language_name (str): language name
            limit (int): limit of suggestions to get

        Returns:
            list of dictionaries with keyword, volume, location and language
        """
        post_data = dict()
        post_data[len(post_data)] = dict(
            keyword=keyword,
            location_name=location_name,
            language_name=language_name,
            include_serp_info=True,
            include_seed_keyword=True,
            limit=limit,
        )
        response = self.post(
            "/v3/dataforseo_labs/google/keyword_suggestions/live", post_data
        )
        if response is None:
            return None
        else:
            keywords = []
            for task in response["tasks"]:
                if "result" in task:
                    for result in task["result"]:
                        if "items" in result:
                            for item in result["items"]:
                                keyword = item["keyword"]
                                search_volume = item["keyword_info"]["search_volume"]
                                keywords.append(
                                    {
                                        "keywordEnriched": keyword,
                                        "keywordLocation": location_name,
                                        "keywordLanguage": language_name,
                                        "keywordVolume": search_volume,
                                        "offerRoot": "KEYWORD_SUGGESTION",
                                    }
                                )
            return keywords

    def get_related_keywords(self, keyword, location_name, language_name, limit=100):
        """Get related keywords for a given keyword, location and language

        Args:
            keyword (str): keyword to get suggestions for
            location_name (str): location name
            language_name (str): language name
            limit (int): limit of suggestions to get

        Returns:
            list of tuples with keyword and search volume"""
        post_data = dict()
        post_data[len(post_data)] = dict(
            keyword=keyword,
            location_name=location_name,
            language_name=language_name,
            limit=limit,
        )
        response = self.post(
            "/v3/dataforseo_labs/google/related_keywords/live", post_data
        )
        if response is None:
            return None
        else:
            keywords = []
            for task in response["tasks"]:
                if "result" in task:
                    for result in task["result"]:
                        if "items" in result:
                            for item in result["items"]:
                                keyword = item["keyword_data"]["keyword"]
                                search_volume = item["keyword_data"]["keyword_info"][
                                    "search_volume"
                                ]
                                keywords.append(
                                    {
                                        "keywordEnriched": keyword,
                                        "keywordLocation": location_name,
                                        "keywordLanguage": language_name,
                                        "keywordVolume": search_volume,
                                        "offerRoot": "RELATED_KEYWORD",
                                    }
                                )

            return keywords
