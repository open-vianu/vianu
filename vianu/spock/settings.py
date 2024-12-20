# General settings
LOGGING_LEVEL = 'DEBUG'
LOGGING_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATA_PATH = "/tmp/spock/"
DATA_FILE = "spock_data"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
N_CHAR_DOC_ID = 12

# Scraping settings
SCRAPING_SOURCES = ['pubmed', 'ema']
MAX_CHUNK_SIZE = 500

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_DB = 'pubmed'
PUBMED_BATCH_SIZE = 500

# NER settings
N_TASKS_DEFAULT = 2
NER_MODELS = ['llama']
MAX_TOKENS = 128.000
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
LLAMA_MODEL='llama3.2'

# UI settings
MAX_JOBS = 5

