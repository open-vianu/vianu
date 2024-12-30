# General settings
LOGGING_LEVEL = 'DEBUG'
LOGGING_FMT = "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"
DATA_PATH = "/tmp/spock/"
DATA_FILE = "spock_data"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
N_CHAR_DOC_ID = 12

# Scraping settings
SCRAPING_SOURCES = ['pubmed', 'ema']
MAX_CHUNK_SIZE = 500
MAX_DOCS_PER_SOURCE = 10
N_SCP_TASKS_DEFAULT = 1

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_DB = 'pubmed'
PUBMED_BATCH_SIZE = 20

# NER settings
N_NER_TASKS_DEFAULT = 1
NER_MODELS = ['llama']
MAX_TOKENS = 128.000
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
LLAMA_MODEL='llama3.2'

# UI settings
MAX_JOBS = 5
UPDATE_INTERVAL = 2

