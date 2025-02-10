import logging
import os

from dotenv import load_dotenv

from vianu import LOG_FMT
from vianu.fraudcrawler.settings import LOG_LEVEL
from vianu.fraudcrawler.src.client import FraudCrawlerClient

logging.basicConfig(level=LOG_LEVEL.upper(), format=LOG_FMT)
logger = logging.getLogger(__name__)


load_dotenv()
_SERPAPI_KEY= os.getenv("SERP_API_TOKEN")
_ZYTEAPI_KEY = os.getenv("ZYTE_API_TOKEN")

# Instantiate the client
client = FraudCrawlerClient(
    serpapi_key=_SERPAPI_KEY,
    zyteapi_key=_ZYTEAPI_KEY,
    location="Switzerland",
)

# Perform search
df = client.run("sildenafil", num_results=10)

# Display results
df
