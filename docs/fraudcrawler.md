
---
title: FraudCrawler
---


# FraudCrawler Documentation

The `FraudCrawler` tool collects URLs suspicious of illegal activity based on medicinal product names.

## Usage
```python
from vianu.fraudcrawler.src.client import FraudcrawlerClient

fc = FraudcrawlerClient()
fc.serpapi_token = "your_serpapi_token"
fc.zyte_api_key = "your_zyte_api_key"

df = fc.search("sildenafil", num_results=5, location="Switzerland")
print(df)
```

### Parameters

- `keyword`: Medicinal product name to search for.
- `num_results`: Number of URLs to retrieve (default: 10).
- `location`: Location filter for URLs.