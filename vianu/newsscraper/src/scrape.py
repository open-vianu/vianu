# scrape.py
import os
import re
import sys
import logging
import time  # Added import for handling sleep
from datetime import datetime
from io import StringIO, BytesIO
from urllib.parse import urljoin, urlparse, quote

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateparser import parse as parse_date
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)


def make_request_with_retries(url, max_retries=5, wait_time=300, **kwargs):
    """
    Makes an HTTP GET request with retry logic for handling 429 status codes.

    Args:
        url (str): The URL to request.
        max_retries (int): Maximum number of retries.
        wait_time (int): Time to wait between retries in seconds.
        **kwargs: Additional arguments to pass to requests.get.

    Returns:
        requests.Response: The HTTP response.

    Raises:
        SystemExit: If maximum retries are exceeded due to persistent 429 responses.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, **kwargs)
            if response.status_code == 429:
                if attempt < max_retries:
                    logging.warning(
                        f"Received 429 Too Many Requests for {url}. Waiting {wait_time} seconds before retrying... (Attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logging.error(
                        f"Received 429 Too Many Requests for {url}. Max retries ({max_retries}) exceeded. Exiting."
                    )
                    sys.exit(1)
            else:
                return response
        except requests.RequestException as e:
            logging.error(f"RequestException for {url}: {e}")
            if attempt < max_retries:
                logging.info(
                    f"Waiting {wait_time} seconds before retrying... (Attempt {attempt}/{max_retries})"
                )
                time.sleep(wait_time)
            else:
                logging.error(f"Max retries exceeded for {url}. Exiting.")
                sys.exit(1)
    # If all retries are exhausted without returning, exit
    logging.error(f"Failed to fetch {url} after {max_retries} attempts.")
    sys.exit(1)


def date_formatter(match):
    """
    Formats German month names to numerical representations and parses the date.

    Args:
        match (re.Match): The regex match object containing the date string.

    Returns:
        str or None: The formatted date in 'YYYY-MM-DD' format or None if parsing fails.
    """
    month_mapping = {
        "Januar": "01",
        "Februar": "02",
        "März": "03",
        "April": "04",
        "Mai": "05",
        "Juni": "06",
        "Juli": "07",
        "August": "08",
        "September": "09",
        "Oktober": "10",
        "November": "11",
        "Dezember": "12",
    }

    date_matched = match.group(0)
    for month_name, month_num in month_mapping.items():
        date_matched = re.sub(month_name, month_num, date_matched, flags=re.I)

    date_matched = re.sub(r"\\n\s+", " ", date_matched)
    date_matched = re.sub(r"-(.*)", "", date_matched)
    date_matched = re.sub(r"(.*),", "", date_matched)
    date_matched = re.sub(r"\s", "", date_matched)
    date_matched = re.sub(r'100%">', "", date_matched)

    parsed_date = parse_date(date_matched, settings={"DATE_ORDER": "DMY"})
    if parsed_date:
        return parsed_date.strftime("%Y-%m-%d")
    else:
        logging.warning(f"Failed to parse date: {date_matched}")
        return None


def convert_pdf_to_txt(fp):
    """
    Converts a PDF file to text.

    Args:
        fp (BytesIO): A file-like object containing PDF data.

    Returns:
        str: The extracted text from the PDF.
    """
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    text = ""
    try:
        for page in PDFPage.get_pages(fp, check_extractable=True):
            interpreter.process_page(page)
        text = retstr.getvalue()
    except Exception as e:
        logging.error(f"Error converting PDF to text: {e}")
    finally:
        device.close()
        retstr.close()

    text = text.replace("\\n", "").lower()
    return text


def fetch_data(url_page, begriff, driver, ziel, sprache):
    """
    Fetches HTML content from a constructed URL and parses it using BeautifulSoup.

    Args:
        url_page (int): The page number to fetch.
        begriff (str): The search term.
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        ziel (str): The target endpoint.
        sprache (str): The language code.

    Returns:
        BeautifulSoup or None: The parsed HTML content or None if fetching fails.
    """
    url = f"https://www.parlament.ch/{sprache}/{ziel}#k={begriff}#s={url_page}"
    logging.info(f"Fetching URL: {url}")
    try:
        driver.get(url)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "ms-srch-item-area"))
        )
        html_source = driver.page_source
        html = BeautifulSoup(html_source, "html.parser")
        logging.debug(f"Fetched HTML content from {url}")
        return html
    except Exception as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return None


def get_dates(urls, driver, sprache, begriff, url_page):
    """
    Extracts dates from a list of URLs by fetching each page and parsing the content.

    Args:
        urls (list): List of URLs to extract dates from.
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        sprache (str): The language code.
        begriff (str): The search term.
        url_page (int): The page number.

    Returns:
        list: A list of formatted dates.
    """
    dates = []
    for url in urls:
        logging.info(f"Extracting date from URL: {url}")
        try:
            driver.get(url)
            element_present = EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".col-sm-8.meta-value.ng-binding")
            )
            WebDriverWait(driver, 30).until(element_present)
            html_source = driver.page_source
            html = BeautifulSoup(html_source, "html.parser")

            divs = html.find_all(
                "div", class_="col-sm-8 meta-value ng-binding"
            ) + html.find_all("td", class_="geschInfo")

            for div in divs:
                match = re.search(r"\d{1,2}\.\d{1,2}\.\d{4}", div.get_text())
                if match:
                    formatted_date = date_formatter(match)
                    if formatted_date:
                        dates.append(formatted_date)
        except TimeoutException:
            logging.warning(f"Timed out waiting for page to load: {url}")
        except Exception as e:
            logging.error(f"Error extracting date from {url}: {e}")
    return dates


def load_data_geschaefte(html, begriff, driver, url_page):
    """
    Parses 'Geschaefte' data from HTML content.

    Args:
        html (BeautifulSoup): The parsed HTML content.
        begriff (str): The search term.
        driver (webdriver.Chrome): The Selenium WebDriver instance.
        url_page (int): The page number.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted data.
    """
    if not html:
        logging.warning("No HTML content provided for Geschaefte data.")
        return pd.DataFrame(columns=["name", "begriff", "date", "url"])

    search_items = html.find_all(
        "div", class_="pd-search-item-type business-item export"
    )
    names = []

    for item in search_items:
        name_tag = item.find("span")
        if name_tag:
            name = re.sub(r"\s+", " ", name_tag.get_text()).strip()
            names.append(name)
        else:
            names.append("")

    search_items_url = html.find_all("h4", class_="ms-srch-item-area")
    urls = []
    for item in search_items_url:
        link = item.find("a", href=True)
        if link:
            urls.append(link["href"])
        else:
            urls.append("")

    urls = [
        urljoin("https://www.parlament.ch", url) if not urlparse(url).scheme else url
        for url in urls
    ]

    dates = []
    if urls:
        logging.info(f"Getting dates for {len(urls)} URLs")
        extracted_dates = get_dates(urls, driver, "de", begriff, url_page)
        dates = extracted_dates

    max_len = max(len(names), len(urls), len(dates))
    names += [""] * (max_len - len(names))
    urls += [""] * (max_len - len(urls))
    dates += [None] * (max_len - len(dates))

    df = pd.DataFrame({"name": names, "begriff": begriff, "date": dates, "url": urls})
    return df


def load_data_news(html, begriff):
    """
    Parses 'News' data from HTML content.

    Args:
        html (BeautifulSoup): The parsed HTML content.
        begriff (str): The search term.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted data.
    """
    if not html:
        logging.warning("No HTML content provided for News data.")
        return pd.DataFrame(columns=["name", "begriff", "date", "url"])

    search_items_name = html.find_all("div", class_="ms-srch-item-excerpt")
    names = []
    for item in search_items_name:
        span = item.find("span")
        if span:
            name = re.sub(r"\s+", " ", span.get_text()).strip()
            names.append(name)
        else:
            names.append("")

    search_items_date = html.find_all(
        "div",
        class_="ms-srch-item-description ms-srch-item-paragraph ms-srch-item-date",
    )
    dates = []
    for item in search_items_date:
        p = item.find("p")
        if p:
            date_str = re.sub(r"\s+", " ", p.get_text()).strip()
            date_str = re.sub(r"-(.*)", "", date_str)
            date_str = re.sub(r"(.*),", "", date_str)
            parsed_date = parse_date(date_str, settings={"DATE_ORDER": "DMY"})
            dates.append(parsed_date.strftime("%Y-%m-%d") if parsed_date else None)
        else:
            dates.append(None)

    search_items_url = html.find_all("h4", class_="ms-srch-item-area")
    urls = []
    for item in search_items_url:
        link = item.find("a", href=True)
        if link:
            urls.append(link["href"])
        else:
            urls.append("")

    urls = [
        urljoin("https://www.parlament.ch", url) if not urlparse(url).scheme else url
        for url in urls
    ]

    df = pd.DataFrame(
        {
            "name": names,
            "begriff": begriff,
            "date": dates,
            "url": urls,
        }
    )
    return df


def load_data_bulletin(html, begriff):
    """
    Parses 'Bulletin' data from HTML content.

    Args:
        html (BeautifulSoup): The parsed HTML content.
        begriff (str): The search term.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted data.
    """
    if not html:
        logging.warning("No HTML content provided for Bulletin data.")
        return pd.DataFrame(columns=["name", "begriff", "date", "url"])

    search_items = html.find_all("div", class_="pd-search-item-type business-item")
    names = []

    for item in search_items:
        name_tag = item.find("span")
        if name_tag:
            name = re.sub(r"\s+", " ", name_tag.get_text()).strip()
            names.append(name)
        else:
            names.append("")

    search_items_date = html.find_all("h4", class_="ms-srch-item-area")
    dates = []
    for item in search_items_date:
        match = re.search(r"\d{2}\.\d{2}\.\d{4}", item.get_text())
        if match:
            date_str = match.group(0)
            parsed_date = parse_date(date_str, settings={"DATE_ORDER": "DMY"})
            dates.append(parsed_date.strftime("%Y-%m-%d") if parsed_date else None)
        else:
            dates.append(None)

    search_items_url = html.find_all("h4", class_="ms-srch-item-area")
    urls = []
    for item in search_items_url:
        link = item.find("a", href=True)
        if link:
            urls.append(link["href"])
        else:
            urls.append("")

    urls = [
        urljoin("https://www.parlament.ch", url) if not urlparse(url).scheme else url
        for url in urls
    ]

    df = pd.DataFrame({"name": names, "begriff": begriff, "date": dates, "url": urls})
    return df


def load_and_fetch_fahne(begriff, driver):
    """
    Fetches and processes 'Fahne' data, including downloading and parsing PDFs.

    Args:
        begriff (str): The search term.
        driver (webdriver.Chrome): The Selenium WebDriver instance.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted data.
    """
    url = f"https://www.parlament.ch/de/suche#Default=%7B%22k%22%3A%22{begriff}%22%2C%22r%22%3A%5B%7B%22n%22%3A%22PdDoctypeDe%22%2C%22t%22%3A%5B%22%5C%22%C7%82%C7%824661686e65%5C%22%22%5D%2C%22o%22%3A%22and%22%2C%22k%22%3Afalse%2C%22m%22%3Anull%7D%5D%7D"
    logging.info(f"Fetching Fahne URL: {url}")
    try:
        driver.get(url)
        WebDriverWait(driver, 30).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "ms-srch-item-excerpt"))
        )
        html_source = driver.page_source
        html = BeautifulSoup(html_source, "html.parser")

        fahne_divs = html.find_all("div", class_="pd-search-item-type news-item pdf")

        names, nos, urls, dates = [], [], [], []
        for div in fahne_divs:
            name = "Fahne"
            link = div.find("a", href=True)
            if link:
                pdf_url = link["href"]
                full_pdf_url = urljoin("https://www.parlament.ch", pdf_url)
                full_pdf_url = quote(full_pdf_url, safe=":/")
                urls.append(full_pdf_url)
            else:
                urls.append("")
                logging.warning("No PDF URL found in Fahne div.")

            if full_pdf_url:
                try:
                    # Use the robust request function here
                    response_pdf = make_request_with_retries(full_pdf_url, timeout=10)
                    if response_pdf.status_code == 200:
                        pdf_stream = BytesIO(response_pdf.content)
                        text = convert_pdf_to_txt(pdf_stream)
                        date_match = re.search(
                            r"e-parl (\d{1,2}\.\d{1,2}\.\d{4})", text
                        )
                        if date_match:
                            formatted_date = date_formatter(date_match)
                            dates.append(
                                formatted_date
                                if formatted_date
                                else datetime.today().strftime("%Y-%m-%d")
                            )
                        else:
                            generic_date_match = re.search(
                                r"\d{1,2}\.\d{1,2}\.\d{4}", text
                            )
                            if generic_date_match:
                                formatted_date = date_formatter(generic_date_match)
                                dates.append(
                                    formatted_date
                                    if formatted_date
                                    else datetime.today().strftime("%Y-%m-%d")
                                )
                            else:
                                dates.append(datetime.today().strftime("%Y-%m-%d"))
                    else:
                        logging.warning(f"Failed to download PDF: {full_pdf_url}")
                        dates.append(datetime.today().strftime("%Y-%m-%d"))
                except Exception as e:
                    logging.error(f"Error fetching PDF from {full_pdf_url}: {e}")
                    dates.append(datetime.today().strftime("%Y-%m-%d"))
            else:
                dates.append(datetime.today().strftime("%Y-%m-%d"))

            name_tags = div.find_all("span")
            for tag in name_tags:
                clean_name = re.sub(r"\s+", " ", tag.get_text()).strip()
                name += f" - {clean_name}"

            no_match = re.search(r" ([0-9]{1,2}\.[0-9]{3})", name)
            if no_match:
                no = no_match.group(1).replace(".", "")
                nos.append(no)
            else:
                nos.append("")

            names.append(name)

        df = pd.DataFrame(
            {"name": names, "begriff": begriff, "date": dates, "url": urls}
        )
        return df

    except TimeoutException:
        logging.warning("Timed out waiting for Fahne page to load.")
        return pd.DataFrame(columns=["name", "begriff", "date", "url"])
    except Exception as e:
        logging.error(f"Error processing Fahne data: {e}")
        return pd.DataFrame(columns=["name", "begriff", "date", "url"])


def start_scraper(
    pages,
    begriffe,
    CHROME_DRIVER_PATH="/usr/local/bin/chromedriver",
    use_local_chrome=False,
):
    """
    Initializes the scraper, processes each search term and page, and stores the data in the database.

    Args:
        pages (list): List of page numbers to scrape.
        begriffe (list): List of search terms.
        CHROME_DRIVER_PATH (str): Path to the ChromeDriver executable.
        use_local_chrome (bool): Whether to use a local Chrome instance.
    """
    if os.getenv("DEPLOYMENT_MODE") == "cloud":
        # Using cloud database:
        cloud_connection = os.getenv("SQLIGHTCONNECTIONSTRING")
        db = DatabaseManager(db_mode="cloud", cloud_connection_string=cloud_connection)
    else:
        # Using the local database
        DATABASE_PATH = "./data/database.db"
        db = DatabaseManager(db_mode="local", local_db_path=DATABASE_PATH)

    options = webdriver.ChromeOptions()
    options.add_argument("--enable-javascript")
    options.add_argument("--no-sandbox")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    logging.info(f"Using ChromeDriver: {CHROME_DRIVER_PATH}")
    service = Service(CHROME_DRIVER_PATH)

    try:
        if use_local_chrome:
            driver = webdriver.Chrome(options=options)
        else:
            driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(5)
        logging.info("Selenium WebDriver initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Selenium WebDriver: {e}")
        sys.exit(1)

    all_data = []
    try:
        for begriff in begriffe:
            logging.info(f"Processing term: {begriff}")
            fahne_df = load_and_fetch_fahne(begriff, driver)
            if fahne_df is not None:
                all_data.append(fahne_df)

            for page in pages:
                logging.info(f"Fetching: Page {page}, Term {begriff}")
                html_news = fetch_data(
                    page, begriff, driver, "services/suche-news", "de"
                )
                html_curia_vista_de = fetch_data(
                    page, begriff, driver, "ratsbetrieb/suche-curia-vista", "de"
                )
                html_curia_vista_fr = fetch_data(
                    page, begriff, driver, "ratsbetrieb/suche-curia-vista", "fr"
                )
                html_curia_vista_it = fetch_data(
                    page, begriff, driver, "ratsbetrieb/suche-curia-vista", "it"
                )
                html_bulletin = fetch_data(
                    page, begriff, driver, "ratsbetrieb/suche-amtliches-bulletin", "de"
                )

                geschaefte_df_de = load_data_geschaefte(
                    html_curia_vista_de, begriff, driver, page
                )
                geschaefte_df_fr = load_data_geschaefte(
                    html_curia_vista_fr, begriff, driver, page
                )
                geschaefte_df_it = load_data_geschaefte(
                    html_curia_vista_it, begriff, driver, page
                )
                news_df = load_data_news(html_news, begriff)
                bulletin_df = load_data_bulletin(html_bulletin, begriff)

                for df in [
                    geschaefte_df_de,
                    geschaefte_df_fr,
                    geschaefte_df_it,
                    news_df,
                    bulletin_df,
                ]:
                    if df is not None and not df.empty:
                        all_data.append(df)
    finally:
        driver.quit()
        logging.info("Selenium WebDriver closed.")

    if all_data:
        # Ensure consistency of columns
        expected_columns = ["name", "begriff", "date", "url"]
        for i, df in enumerate(all_data):
            if set(df.columns) != set(expected_columns):
                logging.warning(
                    f"DataFrame {i} columns mismatch: {df.columns.tolist()}"
                )
                df = df.reindex(columns=expected_columns, fill_value=None)

        final_df = pd.concat(all_data, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=["name", "begriff", "date", "url"])

    # Current date in the desired format
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Replace None, empty strings, or NaN with today's date
    final_df["date"] = final_df["date"].replace([None, ""], pd.NaT)
    final_df["date"] = final_df["date"].fillna(today_date)

    logging.info(f"Total records fetched: {len(final_df)}")

    inserted_count = db.insert_articles(final_df)
    logging.info(f"Inserted {inserted_count} new records into the database.")
    logging.info("Scraping completed successfully!")


def main(use_local_chrome=False):
    """
    The main entry point of the scraper. Cleans the temp folder and starts the scraping process.

    Args:
        use_local_chrome (bool): Whether to use a local Chrome instance.
    """
    pages = [1]
    begriffe = ["covid"]
    start_scraper(pages, begriffe, use_local_chrome=use_local_chrome)


if __name__ == "__main__":
    main(use_local_chrome=True)
