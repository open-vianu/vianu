# General settings
LOGGING_LEVEL = 'DEBUG'
LOGGING_FMT = "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s | %(message)s"
DATA_PATH = "/tmp/spock/"
DATA_FILE = "spock_data"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
N_CHAR_DOC_ID = 12
GRADIO_SERVER_PORT=7868

# Scraping settings
SCRAPING_SOURCES = ['pubmed', 'ema', 'mhra']
MAX_CHUNK_SIZE = 500
MAX_DOCS_SRC = 5
N_SCP_TASKS_DEFAULT = 1

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_DB = 'pubmed'
PUBMED_BATCH_SIZE = 20

# NER settings
N_NER_TASKS_DEFAULT = 1
LARGE_LANGUAGE_MODELS = ['llama', 'openai']
MAX_TOKENS = 128.000
LLAMA_MODEL='llama3.2'
OPENAI_MODEL='gpt-4o'

# UI settings
MAX_JOBS = 5
UPDATE_INTERVAL = 2

