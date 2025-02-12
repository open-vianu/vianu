import asyncio
import logging
import os

from dotenv import load_dotenv

from vianu import LOG_FMT
from vianu.fraudcrawler.settings import LOG_LEVEL
from vianu.fraudcrawler.src.client import FraudCrawlerClient

logging.basicConfig(level=LOG_LEVEL.upper(), format=LOG_FMT)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
_SERPAPI_KEY = os.getenv("SERPAPI_KEY")
_ZYTEAPI_KEY = os.getenv("ZYTEAPI_KEY")

# Instantiate the client
client = FraudCrawlerClient(
    serpapi_key=_SERPAPI_KEY,
    zyteapi_key=_ZYTEAPI_KEY,
    location="Switzerland",
)

# Perform sequential search
# df = client.run("sildenafil", num_results=10) # Sequential
# print(df.head())

# Perform async search
asyncio.run(client.async_run("sildenafil", num_results=10))  # Async
