DOC_MAPPING = {
    "doc_01": {
        "name": "M7(R2) Addendum: Application of the Principles of the ICH M7 Guideline to Calculation of Compound-Specific Acceptable Intakes",
        "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m7r2-addendum-application-principles-ich-m7-guideline-calculation-compound-specific-acceptable",
    },
    "doc_02": {
        "name": "Q3C(R8) Impurities: Guidance for Residual Solvents Guidance for Industry",
        "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/q3cr8-impurities-guidance-residual-solvents-guidance-industry",
    },
    "doc_03": {
        "name": "Q3C Impurities: Guidance for Residual Solvents Guidance for Industry - Tables and List Rev. 4",
        "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/q3c-tables-and-list-rev-4",
    },
    "doc_04": {
        "name": "Q3D(R2) – Guideline for Elemental Impurities",
        "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/q3dr2-guideline-elemental-impurities",
    },
}

PATH_TO_PDF_IMAGES = "data/001_pdfs_to_images/"
PATH_TO_PDF_CONTENT = "data/003_images_to_content/"
PATH_TO_PDF_CHUNKS = "data/005_chunked_content/"
PATH_TO_INDEX = "data/007_chunks_index/"
PATH_TO_ANNOTATIONS = "data/008_annotations/"
PATH_TO_HUMAN_ANNOTATIONS = "data/008_annotations/human_annotations.xlsx"
PATH_TO_EVALUATION = "data/009_evaluation/"

# content type to remove while chunking the content
NOT_ALLOWED_CONTENT_TYPE = ["page-header", "page-footer", "footnote"]
# metadata to add to the above content types while chunking the content
METADATA_FIELDS = ["section", "subsection"]

# page classes to remove while indexing the chunks
NOT_ALLOWED_CLASSES = ["LEGAL_DISCLAIMERS", "ACKNOWLEDGMENTS", "COVER_PAGE", "CITATIONS_REFERENCES"]

# keys to consider while embedding the chunks
EMBEDDING_KEYS = ["section", "subsection", "name", "caption", "content"]

# https://huggingface.co/dunzhang/stella_en_400M_v5
LOCAL_EMBEDDING_MODEL = "dunzhang/stella_en_400M_v5"
# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
LOCAL_LLM = "Llama-3.2-1B-Instruct-Q8_0"

INDEX_FILENAME = "faiss_index"
INDEX_NAMES = ["chunks"]

CHUNK_TYPES_TO_ANNOTATE = ["paragraph", "table", "list"]
PAGE_CLASSES_TO_ANNOTATE = ["MAIN_CONTENT", "TABLES"]
MIN_CONTENT_LENGTH_TO_ANNOTATE = 100
MAX_CHUNKS_TO_ANNOTATE_BY_DOC = 100

RETRIEVAL_TOP_K_TO_EVALUATE = 30
GENERATION_TOP_K_TO_EVALUATE = 5
