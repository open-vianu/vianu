# db_setup.py
import sqlite3
import os


def setup_db(DB_PATH="/app/data/database.db"):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS search_terms (
        Begriffe TEXT UNIQUE
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY,
        name TEXT,
        begriff TEXT,
        date TEXT,
        url TEXT,
        UNIQUE(name, begriff)
    )
    """)

    # Create metadata table
    c.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        last_scraping_timestamp TEXT,
        last_scraping_num_articles INTEGER,
        last_scraping_num_search_terms INTEGER,
        last_scraping_status TEXT
    )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


if __name__ == "__main__":
    setup_db(DB_PATH=os.getenv("LOCAL_DB_PATH", "/app/data/database.db"))
