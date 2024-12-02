from typing import List
import comparison_utils


def match_seq_against_list(
    data_matched: List,
    database: List,
    name_source: str,
    searched_word,
    threshold: float,
):
    """
    Function goes through each database comparing word of interest "searched_word" with every element and collects only
    those that stay above a given threshold.
    """
    for each in database:
        ratio = comparison_utils.orthographic_comparison(
            searched_word, each, "SequenceMatcher"
        )

        ratio_phonetic = comparison_utils.phonetic_comparison(
            searched_word, each, "Kolner"
        )

        ratio_fr_phonetic_fonem = comparison_utils.phonetic_comparison(
            searched_word, each, "fr-fonem"
        )

        # THIS FUNCTION ADDED AN ALTERNATIVE FRENCH PHONETIC COMPARISON - DISCONTINUED IN MARCH 2022.
        # ratio_fr_phonetic_henry = phonetic_comparison(searched_word, each, 'fr-henry')

        ratio_combined = (ratio + ratio_phonetic) / 2
        if ratio_combined >= (float(threshold) / 100):
            data_matched.append(
                [
                    each,
                    round(ratio * 100),
                    round(ratio_phonetic * 100),
                    round(ratio_fr_phonetic_fonem * 100),
                    name_source,
                ]
            )

    return data_matched
