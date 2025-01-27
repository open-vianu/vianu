# General settings
LOG_LEVEL = "DEBUG"
N_CHAR_DOC_ID = 12
FILE_PATH = "/tmp/spock/"  # nosec
FILE_NAME = "spock"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Gradio app settings
GRADIO_APP_NAME = "SpoCK"
GRADIO_SERVER_PORT = 7868
GRADIO_MAX_JOBS = 5
GRADIO_UPDATE_INTERVAL = 2

# Scraping settings
SCRAPING_SOURCES = ["pubmed", "ema", "mhra", "fda"]
MAX_CHUNK_SIZE = 500
MAX_DOCS_SRC = 5
N_SCP_TASKS = 2

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_DB = "pubmed"
PUBMED_BATCH_SIZE = 20

# NER settings
LARGE_LANGUAGE_MODELS = ["openai", "llama"]
MAX_TOKENS = 128.000
LLAMA_MODEL = "llama3.2"
OPENAI_MODEL = "gpt-4o"
N_NER_TASKS = 2
OLLAMA_BASE_URL_ENV_NAME = "OLLAMA_BASE_URL"
OPENAI_API_KEY_ENV_NAME = "OPENAI_API_KEY"
MODEL_TEST_QUESTION = "Are you available?"
