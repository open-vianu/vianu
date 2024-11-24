import re
import pandas as pd
from typing import List
from .config import (
    FDA_FILEPATH,
    USAN_FILEPATH,
    RXNORM_FILEPATH,
    SWISSMEDIC_FILEPATH,
)


def extract_drug_name(text):
    split_char = ","
    compound_raw = text.split(split_char)[0]
    pattern = r" [0-9]*"
    compound_wo_alone_numbers = re.sub(pattern, " ", compound_raw)
    pattern2 = r"/[0-9]*"
    compound_number_filtered = re.sub(pattern2, " ", compound_wo_alone_numbers)

    stopwords = [
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
    ]

    words_within_medicament = compound_number_filtered.split()
    result_words = [
        word for word in words_within_medicament if word.lower() not in stopwords
    ]
    compound_filtered = " ".join(result_words)

    if compound_filtered[0] == "-":
        compound_filtered = compound_filtered[1:]

    return compound_filtered


def read_medicament_file_as_list(chosen_sources: List):
    """
    read_medicament_file reads files from any of the four sources ['fda','rxnorm','usan','swissmedic']

    Input:
    chosen_sources: List with any number out of the possible sources ['fda','rxnorm','usan','swissmedic']

    Return:
    File: read individual file for that specific source.
    """
    if chosen_sources == "fda":
        # Read file for comparison
        filepath = FDA_FILEPATH
        data = pd.read_csv(filepath, header=None, delimiter="|")
        name = pd.Series(data.iloc[:, [1, 4]].to_numpy().flatten())
        name = name.drop_duplicates()
        name = name.dropna().to_list()

    elif chosen_sources == "usan":
        filepath = USAN_FILEPATH
        data = pd.read_csv(filepath, header="infer", delimiter=";")
        name = data.Examples.to_numpy().flatten()
        name = pd.Series(name).drop_duplicates()
        name = name.dropna()
        name = name.to_list()
        print("Number of drugs USAN database: ", len(name))

    elif chosen_sources == "rxnorm":
        filepath = RXNORM_FILEPATH
        data = pd.read_csv(filepath, header=None, delimiter="|")
        name = data.iloc[:, 1].to_numpy().flatten()
        name = pd.Series(name).drop_duplicates()
        name = name.dropna().to_numpy()
        list_output = [drug.upper() for drug in name]
        print("Number of drugs at the RxNorm database: ", len(list_output))
        return list_output

    elif chosen_sources == "swissmedic":
        filepath = SWISSMEDIC_FILEPATH
        data = pd.read_excel(filepath, skiprows=6, header=0)
        col_compounds = "Bezeichnung des Arzneimittels\n\n\nDénomination du médicament"

        data["drug_name"] = data[col_compounds].apply(extract_drug_name)
        name = data["drug_name"].to_numpy().flatten()
        name = pd.Series(name).drop_duplicates()
        name = name.dropna().to_numpy()
        list_output = [drug.upper() for drug in name]
        print("Number of drugs at the Swissmedic database: ", len(list_output))
        return list_output

    else:
        name = False

    return name
