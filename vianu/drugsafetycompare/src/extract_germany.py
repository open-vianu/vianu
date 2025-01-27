"""
Module for extracting drug information from the German Rote Liste website.

This module provides a class `GermanDrugInfoExtractor` that allows searching for drugs
and retrieving their undesired effects (Nebenwirkungen) from the Rote Liste website.

Usage:
    extractor = GermanDrugInfoExtractor()
    products = extractor.search_drug('aspirin')
    side_effects = extractor.get_undesired_effects(products[0]['link'])
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from typing import List, Dict


class GermanDrugInfoExtractor:
    """
    A class to extract drug information from the German Rote Liste website.

    Attributes:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
    """

    def __init__(self):
        """
        Initializes the GermanDrugInfoExtractor with Selenium WebDriver.
        """
        options = Options()
        options.add_argument("--headless")  # Run in headless mode.
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(options=options)

    def search_drug(self, drug_name: str) -> List[Dict[str, str]]:
        """
        Searches for products matching the drug name on the Rote Liste website.

        Args:
            drug_name (str): The name of the drug to search for.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing product names and links.
        """
        url = f"https://www.rote-liste.de/suche/{drug_name}"
        self.driver.get(url)
        time.sleep(3)  # Wait for the page to load

        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        # Find products
        products = soup.find_all("div", class_="row py-3 px-4 ml-3 ml-lg-0")

        product_list = []

        for product in products:
            a_tag = product.find("a")
            if a_tag:
                name = a_tag.get_text(strip=True)
                link = a_tag["href"]
                if not link.startswith("http"):
                    link = "https://www.rote-liste.de" + link
                product_list.append({"name": name, "link": link})

        return product_list

    def get_undesired_effects(self, product_url: str) -> str:
        """
        Retrieves the undesired effects (Nebenwirkungen) from the product page.

        Args:
            product_url (str): The URL of the product page.

        Returns:
            str: The undesired effects text if found, otherwise an empty string.
        """
        self.driver.get(product_url)
        time.sleep(3)  # Wait for the page to load

        product_soup = BeautifulSoup(self.driver.page_source, "html.parser")

        # Find the 'Nebenwirkungen' section
        nebenwirkungen_heading = product_soup.find("div", id="nebenwirkungen")
        if nebenwirkungen_heading:
            # The content is in the next sibling div with class 'product-detail--box-subtitle'
            content_div = nebenwirkungen_heading.find_next_sibling(
                "div", class_="product-detail--box-subtitle"
            )
            if content_div:
                return content_div.get_text(strip=True)
        return "No 'Nebenwirkungen' section found."

    def quit(self):
        """
        Closes the Selenium WebDriver.
        """
        self.driver.quit()
