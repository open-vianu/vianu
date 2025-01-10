---
title: Getting started
layout: home
nav_order: 1
---

# Welcome to VIANU!

# Vianu is a package meant to provide support to devs in the life sciences

[Get Started](#getting-started){: .btn .btn-purple }

[Vianu] is a *Python package* designed for developers working in the life sciences and healthcare sectors. 
It provides access to a variety of tools and workflows, allowing users to quickly build, validate, and deploy 
data-driven applications.

<h2 id="getting-started">Getting Started</h2>

This is how you install all of our cool ninja moves

## Available Tools

- **FraudCrawler**: A data ingestion and transformation pipeline for real-world healthcare data.
- **Lasa**: A tool for phonetic comparison of novel drug names with authorized ones from different locations.
- **Spock**: Perform literature research for adverse effects.
- **Ragulator**: Conduct semantic searches in official guidelines.
- 

## Prerequisites
- Python 3.11 or higher
- Internet connection for API calls

## Quick Installation
```bash
pip install vianu==0.1.2
```

### Example usage

```python
from vianu.fraudcrawler.src.client import FraudcrawlerClient

fc = FraudcrawlerClient()
fc.serpapi_token = "your_token"
fc.zyte_api_key = "your_key"

df = fc.search("sildenafil", num_results=5, location="Switzerland")
print(df)
```

Other than that, go ahead and `have fun` with this repository!

[Vianu]: https://github.com/open-vianu/vianu
