import os

from dotenv import load_dotenv
from vianu.fraudcrawler.src.client import FraudCrawlerClient

load_dotenv()

# Instantiate the client
nc_client = FraudCrawlerClient()

# Set API tokens
nc_client.serpapi_token = os.getenv("SERP_API_TOKEN", "YOUR_SERPAPI_TOKEN")
nc_client.zyte_api_key = os.getenv("ZYTE_API_TOKEN", "YOUR_ZYTE_API_KEY")

# Perform search
df = nc_client.search("sildenafil", num_results=5, location="Switzerland")

# Display results
df
