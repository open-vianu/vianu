# General settings
LOG_LEVEL = "DEBUG"
PROCESSES = 5

# Gradio app settings
GRADIO_SERVER_PORT = 7863
GRADIO_APP_NAME = "LASA"
GRADIO_RESULTS_COLUMNS = ["Phonetic", "Spelling", "Combined"]
GRADIO_RESULTS_COLUMNS_DEFAULT = "combined"

# SMC settings
SMC_FILENAME = "swissmedic_20241231.xlsx"
SMC_NON_DRUGS = [
    "a",
    "b",
    "c",
    "g",
    "n",
    "p",
    "s",
    "u",
    "e.",
    "ie",
    "%",
    "i.e.",
    ".0",
    ".05",
    ".1",
    ".2",
    ".3",
    ".4",
    ".5",
    ".6",
    ".625",
    ".7",
    ".8",
    ".9",
    ".25",
    ".75",
    "9",
    "10",
    "15",
    "20",
    "25",
    "30",
    "35",
    "43",
    "50",
    "52",
    "160",
    "200",
    "240",
    "300",
    "320",
    "400",
    "800",
    ".1%",
    ".25%",
    ".3%",
    ".5%",
    "10%",
    "15%",
    "20%",
    "'000",
    "'250",
    ".5mg",
    "mcg",
    "mg",
    "ΜG",
    "ml",
    "mmol",
    "l",
    "ca",
    "xr",
    "ug",
    "hg",
    "cu",
    "qu",
    "arg.",
    "c.",
    "mikrogramm",
    "liquid",
    "gefärbt",
    "ungefärbt",
    "agenti",
    "conservanti",
    "nr.",
    "adultes",
    "enfants",
    "zum",
    "einnehmen",
    "für",
    "erwachsene",
    "und",
    "kinder",
    "ab",
    "jahren",
    "äusserlich",
    "con",
    "spezifiziert",
    "spécifié",
    "extract",
    "preparation",
    "tabletten",
    "pulver",
    "zur",
    "herstellung",
    "einnehmen",
    "sandoz",
    "mundipharma",
    "spirig",
    "hc",
    "maddox",
    "bayer",
    "nobel",
    "zentiva",
    "accord",
    "labatec",
    "ideogen",
    "viatris",
    "fresenius",
    "pfizer",
    "coop",
    "vitality",
    "sun",
    "store",
    "axapharm",
    "orpha",
    "eco",
]

# FDA settings
FDA_FILENAME = "fda_products_20250121.csv"

# LASA settings
SOURCES = ["swissmedic", "fda"]
THRESHOLD = 70
