import os

from dotenv import load_dotenv
from vianu.fraudcrawler.src.crawler import Crawler

load_dotenv()
_SERPAPI_KEY= os.getenv("SERP_API_TOKEN")
_ZYTEAPI_KEY = os.getenv("ZYTE_API_TOKEN")

# Instantiate the client
crawler = Crawler(
    serpapi_key=_SERPAPI_KEY,
    zyteapi_key=_ZYTEAPI_KEY,
    location="Switzerland",
)

# Perform search
df = crawler.apply("sildenafil", num_results=10)

# Display results
df
