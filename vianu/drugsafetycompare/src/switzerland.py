"""
Module for extracting drug information from the Swiss SAI Refdata website.

This module provides a class `SwissDrugInfoExtractor` that allows searching for drugs
and retrieving their side effects from the SAI Refdata website.

Usage:
    extractor = SwissDrugInfoExtractor()
    products = extractor.search_drug('aspirin')
    side_effects = extractor.get_side_effects(products[0]['link'])
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict
import urllib.parse
import time
import re
from bs4 import BeautifulSoup


class SwissDrugInfoExtractor:
    """
    A class to extract drug information from the Swiss SAI Refdata website.

    Attributes:
        driver (webdriver.Chrome): The Selenium WebDriver instance.
    """

    def __init__(self):
        """
        Initializes the SwissDrugInfoExtractor with Selenium WebDriver.
        """
        options = Options()
        # For debugging, you can comment out the headless option
        options.add_argument("--headless")  # Run in headless mode.
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        # Enable JavaScript
        options.add_argument("--enable-javascript")

        self.driver = webdriver.Chrome(options=options)

    def search_drug(self, drug_name: str) -> List[Dict[str, str]]:
        """
        Searches for products matching the drug name on the SAI Refdata website.

        Args:
            drug_name (str): The name of the drug to search for.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing product names and links.
        """
        # Use the German version of the site
        encoded_drug_name = urllib.parse.quote(drug_name)
        url = f"https://sai.refdata.ch/de?q={encoded_drug_name}&isfv=false&vw=ham&sg=default"
        self.driver.get(url)

        # Wait until the table with search results is present
        wait = WebDriverWait(self.driver, 10)
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table")))
        except Exception as e:
            print(f"An error occurred while waiting for the results: {e}")
            return []

        # Now find the rows in the table body
        rows = self.driver.find_elements(By.CSS_SELECTOR, "table.table tbody tr")

        product_list = []
        for row in rows:
            try:
                # The product name is in the third cell (index 2)
                cells = row.find_elements(By.TAG_NAME, "td")
                name = cells[0].text.strip()
                link = row.get_attribute("data-url")
                if link and not link.startswith("http"):
                    link = "https://sai.refdata.ch" + link
                product_list.append({"name": name, "link": link})
            except Exception as e:
                print(f"Error processing row: {e}")

        return product_list

    def get_side_effects(self, product_url: str) -> str:
        """
        Retrieves the side effects from the product details.

        Args:
            product_url (str): The URL of the product details page.

        Returns:
            str: The side effects text if found, otherwise a message indicating it's not found.
        """
        self.driver.get(product_url)
        wait = WebDriverWait(self.driver, 30)  # Increased timeout

        # Click the "Fachinformation" button
        try:
            # Wait for the page to load completely
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # Locate the 'Fachinformation' link by href attribute
            fachinfo_button = self.driver.find_element(
                By.XPATH, "//a[contains(@href, 'textType=FI')]"
            )
            fachinfo_button.click()
        except Exception as e:
            return f"Could not find or click the 'Fachinformation' button: {e}"

        # Switch to the new window/tab
        time.sleep(2)  # Wait for the new window to open
        windows = self.driver.window_handles
        if len(windows) > 1:
            self.driver.switch_to.window(windows[1])
        else:
            return "No new window opened after clicking 'Fachinformation'."

        # Wait for the new page to load
        try:
            # Wait for the article content to be populated
            wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "article.monographie-content")
                )
            )
            # Wait until the shadow DOM is attached
            wait.until(
                lambda driver: driver.execute_script(
                    "return document.querySelector('article.monographie-content').shadowRoot != null"
                )
            )
            # Now wait until the shadow DOM has content
            wait.until(
                lambda driver: driver.execute_script(
                    "return document.querySelector('article.monographie-content').shadowRoot.innerHTML.length > 0"
                )
            )
        except Exception as e:
            return f"An error occurred while waiting for the Fachinformation page to load: {e}"

        # Get the content from the shadow DOM
        try:
            shadow_root_content = self.driver.execute_script(
                "return document.querySelector('article.monographie-content').shadowRoot.innerHTML"
            )

            if not shadow_root_content:
                return "Shadow DOM content not found."

            # Parse the HTML content
            soup = BeautifulSoup(shadow_root_content, "html.parser")

            # Debugging: Print the soup content (optional)
            # print(soup.prettify())

            # Find the 'Unerwünschte Wirkungen' heading
            side_effects_heading = soup.find(
                "p", string=re.compile(r"Unerwünschte Wirkungen", re.IGNORECASE)
            )

            if side_effects_heading:
                # Initialize an empty list to hold the side effects paragraphs
                side_effects_text = []

                # Start with the next sibling
                next_node = side_effects_heading.find_next_sibling()

                # Loop through siblings until the next section heading
                while next_node:
                    # Check if this node is a section heading
                    if next_node.name == "p" and next_node.get("id", "").startswith(
                        "section"
                    ):
                        # Reached the next section
                        break
                    # Otherwise, add the text to the list
                    side_effects_text.append(
                        next_node.get_text(separator=" ", strip=True)
                    )
                    next_node = next_node.find_next_sibling()

                side_effects_text = "\n".join(side_effects_text).strip()
                if side_effects_text:
                    return side_effects_text
                else:
                    return "No side effects content found under the heading."
            else:
                return "No 'Unerwünschte Wirkungen' section found."
        except Exception as e:
            return f"An error occurred while extracting side effects: {e}"
        finally:
            # Close the new window and switch back to the original
            self.driver.close()
            self.driver.switch_to.window(windows[0])

    def quit(self):
        """
        Closes the Selenium WebDriver.
        """
        self.driver.quit()


if __name__ == "__main__":
    extractor = SwissDrugInfoExtractor()

    try:
        # Define the drug name
        drug_name = "aspirin"

        # Search for products matching the drug name
        products = extractor.search_drug(drug_name)

        if len(products) >= 3:
            # Select the third product (index 2)
            selected_product = products[2]

            # Get the product URL
            product_url = selected_product["link"]

            # Retrieve side effects
            side_effects = extractor.get_side_effects(product_url)

            # Print the results
            print(f"Selected Product: {selected_product['name']}\n")
            print("Side Effects:")
            print(side_effects)
        else:
            print("Less than 3 products found.")
            print(f"Found {len(products)} products.")
            for product in products:
                print(product)

    finally:
        # Close the extractor
        extractor.quit()
