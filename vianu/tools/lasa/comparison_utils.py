import importlib
from typing import List
from difflib import SequenceMatcher
from abydos.phonetic import fonem, henry_early
cologne = importlib.import_module("cologne_phonetics")


def orthographic_comparison(searched_word: str, element_list: List, option: str) -> float:
    """
    This function takes a "searched_string" which is the name that we interested in and compares it
    against the "element_list" which are the names that are pulled from the large datasets. Short said,
    we are interested for example in the name "MAVIX", and compare again the entire list one by one

    :param searched_word: medicament name that we are interested in
    :param element_list: medicament names from the databases
    :param option: given phonetic technic for comparison
    :return: phonetic_score: which is the estimation of the phonetic matching
    """
    if option == 'SequenceMatcher':
        ratio = SequenceMatcher(None, searched_word, element_list).ratio()
    return ratio


def phonetic_comparison(searched_word, element_list, option):
    """
    drug_testname = searched_string
    drug_found = element_list

    This function takes a "searched_string" which is the name that we interested in and compares it
    against the "element_list" which are the names that are pulled from the large datasets. Short said,
    we are interested for example in the name "MAVIX", and compare again the entire list one by one

    :param searched_word: medicament name that we are interested in
    :param element_list: medicament names from the databases
    :param option: given phonetic technic for comparison
    :return: phonetic_score: which is the estimation of the phonetic matching
    """
    ratio = -1
    if option == 'Kolner':
        drug_interest = cologne.encode(searched_word.upper())[0][1]
        drug_in_our_list = cologne.encode(element_list.upper())[0][1]
        ratio = SequenceMatcher(None, drug_interest, drug_in_our_list).ratio()

        phonetic_score = float(round(ratio, 2))

    if option == 'fr-fonem':
        drug_interest = fonem(searched_word.upper())
        drug_in_our_list = fonem(element_list.upper())
        ratio = SequenceMatcher(None, drug_interest, drug_in_our_list).ratio()
        phonetic_score = float(round(ratio, 2))

    if option == 'fr-henry':
        drug_interest = henry_early(searched_word.upper())
        drug_in_our_list = henry_early(element_list.upper())
        ratio = SequenceMatcher(None, drug_interest, drug_in_our_list).ratio()
        phonetic_score = float(round(ratio, 2))

    return phonetic_score