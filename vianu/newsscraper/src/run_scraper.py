# run_scraper.py
import os
import logging
from datetime import datetime
from scrape import start_scraper
from db_manager import DatabaseManager


def run_and_update_metadata():
    if os.getenv("DEPLOYMENT_MODE") == "cloud":
        # Using cloud database:
        cloud_connection = os.getenv("SQLIGHTCONNECTIONSTRING")
        db = DatabaseManager(db_mode="cloud", cloud_connection_string=cloud_connection)
    else:
        # Using the local database
        DATABASE_PATH = "./data/database.db"
        db = DatabaseManager(db_mode="local", local_db_path=DATABASE_PATH)
    logging.info("Starting automated scraping process.")

    search_terms = db.list_terms()
    num_search_terms = len(search_terms)
    status = "Success"
    inserted_articles = 0

    try:
        pages = [1, 2]
        start_scraper(pages, search_terms)

        total_articles = db.count_articles()
        inserted_articles = total_articles
    except Exception as e:
        logging.error(f"Error during scraping: {e}")
        status = f"Failed: {e}"

    end_time = datetime.now()
    timestamp = end_time.strftime("%Y-%m-%d %H:%M:%S")

    db.insert_metadata(timestamp, inserted_articles, num_search_terms, status)
    logging.info("Automated scraping process completed.")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    run_and_update_metadata()
