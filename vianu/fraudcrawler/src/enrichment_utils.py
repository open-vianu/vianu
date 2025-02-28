import pandas as pd


def filter_keywords(keyword):
    """Filter irrelevant keywords based on a list for 4 languages

    Args:
        keyword (str): keyword to be filtered

    Returns:
        keyword (str)s: Return None if the keyword is blacklisted, otherwise return the keyword
    """

    blacklist_german = [
        "nebenwirkung",
        "erfahrung",
        "gefährlich",
        "gefahr",
        "risiko",
        "bewertung",
        "bericht",
        "warnung",
        "symptome",
        "kritik",
    ]
    blacklist_english = [
        "side effect",
        "dangerous",
        "danger",
        "risk",
        "report",
        "warning",
        "symptom",
        "criticism",
    ]
    blacklist_french = [
        "expérience",
        "dangereux",
        "danger",
        "risque",
        "rapport",
        "avertissement",
        "symptômes",
        "secondaire",
        "critique",
    ]
    blacklist_italian = [
        "collateral",
        "pericolo",
        "rischio",
        "recensione",
        "rapporto",
        "avvertimento",
        "sintomi",
        "critica",
    ]

    blacklist = (
        blacklist_german + blacklist_english + blacklist_french + blacklist_italian
    )

    if not any(blacklisted_word in keyword for blacklisted_word in blacklist):
        return None
    else:
        return keyword


def aggregate_keywords(keywords):
    """Aggregate keywords based on their search volume, same keyword for different language and location
    will be aggregated by adding their volume, while keeping location and language information.

    Args:
        keywords: list of dictionaries with keyword, volume, location, and language.

    Returns:
        df_agg: pandas dataframe with aggregated keywords and other columns preserved."""

    df = pd.DataFrame(keywords)

    # Group by 'keyword' and aggregate 'volume' using sum, keeping the first value of 'location' and 'language'
    df_agg = (
        df.groupby("keywordEnriched")
        .agg(
            {
                "keywordVolume": "sum",
                "keywordLocation": "first",
                "keywordLanguage": "first",
                "offerRoot": "first",
            }
        )
        .reset_index()
    )

    # Sort by volume in descending order
    df_agg = df_agg.sort_values(by="keywordVolume", ascending=False)

    return df_agg


def estimate_volume_per_url(
    urls, keyword_search_volume, keyword, keywordLocation, keywordLanguage, offerRoot
):
    """Estimate the volume per url based on the rank click shares (
    https://www.advancedwebranking.com/ctrstudy/)

    Args:
        urls: list of urls
        keyword_search_volume: search volume of the keyword

    Returns:
        results: list of dictionaries with url and estimated volume per url
    """
    rank_click_shares = [
        0.33,
        0.17,
        0.11,
        0.08,
        0.06,
        0.05,
        0.04,
        0.035,
        0.03,
        0.025,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.005,
        0.005,
        0.0034,
        0.0033,
        0.0033,
    ]
    volumes_per_url = [round(x * keyword_search_volume, 0) for x in rank_click_shares]
    results = [
        {
            "url": url,
            "keywordVolume": vol,
            "keywordEnriched": keyword,
            "keywordLanguage": keywordLanguage,
            "keywordLocation": keywordLocation,
            "offerRoot": offerRoot,
        }
        for (url, vol) in zip(urls[0:20], volumes_per_url)
    ]
    return results


def aggregate_urls(urls):
    """Aggregate urls based on their volume. Same urls will be aggregated by adding their volume

    Args:
        urls: list of dictionaries with url and volume

    Returns:
        df_agg: pandas dataframe with aggregated urls"""
    df = pd.DataFrame(urls)
    df_agg = (
        df.groupby("url")
        .agg(
            {
                "keywordVolume": "sum",
                "keywordEnriched": "first",
                "keywordLanguage": "first",
                "offerRoot": "first",
            }
        )
        .reset_index()
    )
    df_agg = df_agg.sort_values(by="keywordVolume", ascending=False)
    return df_agg.to_dict(orient="records")
