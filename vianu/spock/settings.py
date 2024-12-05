# General settings
DEFAULT_LOGGING_LEVEL = 'INFO'
LOGGING_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DEFAULT_DATA_DUMP = "/tmp/spock/data.json"

# Scraping settings
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_DB = 'pubmed'
PUBMED_BATCH_SIZE = 500

# NER settings
DEFAULT_MAX_TOKENS = 128.000
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_LLAMA_MODEL='llama3.2'
