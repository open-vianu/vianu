# db_manager.py
import os
import logging
import sqlite3
import sqlitecloud
import hashlib


class DatabaseManager:
    def __init__(
        self,
        local_db_path=os.getenv("LOCAL_DB_PATH", "./data/database.db"),
        db_mode=os.getenv("DEPLOYMENT_MODE", "local"),
        cloud_connection_string=None,
        cloud_db_name="articles",
    ):
        """
        Initialize the DatabaseManager.

        :param local_db_path: Path to local SQLite database file (used if db_mode='local').
        :param db_mode: "local" or "cloud". If "cloud", use the cloud_connection_string and cloud_db_name.
        :param cloud_connection_string: Connection string to a remote (cloud) SQLite database.
        :param cloud_db_name: Database name to USE after connecting in cloud mode.
        """

        self.db_mode = db_mode
        self.local_db_path = local_db_path
        self.cloud_connection_string = cloud_connection_string
        self.cloud_db_name = cloud_db_name

        logging.info(f"Initializing database with mode: {self.db_mode}")

        # Initialize the database (create tables if needed)
        if self.db_mode == "local":
            self._ensure_db_dir_exists()
            self.initialize_database()

    def _ensure_db_dir_exists(self):
        os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)

    def _connect(self):
        if self.db_mode == "local":
            # Connect to local SQLite file
            try:
                conn = sqlite3.connect(self.local_db_path)
                return conn
            except sqlite3.Error as e:
                logging.error(f"Database connection failed (local): {e}")
                return None
        else:
            # Connect to cloud SQLite database
            if not self.cloud_connection_string:
                logging.error("No cloud connection string provided for cloud mode.")
                return None
            if not self.cloud_db_name:
                logging.error(
                    "No cloud_db_name provided for cloud mode. Cannot run USE DATABASE."
                )
                return None
            try:
                conn = sqlitecloud.connect(self.cloud_connection_string)
                # Now select the database
                conn.execute(f"USE DATABASE {self.cloud_db_name}")
                return conn
            except Exception as e:
                logging.error(f"Database connection failed (cloud): {e}")
                return None

    def initialize_database(self):
        """Create required tables if they don't exist."""
        conn = self._connect()
        if not conn:
            return

        create_articles_table = """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            begriff TEXT,
            date TEXT,
            url TEXT
        )
        """

        create_search_terms_table = """
        CREATE TABLE IF NOT EXISTS search_terms (
            Begriffe TEXT PRIMARY KEY
        )
        """

        create_metadata_table = """
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_scraping_timestamp TEXT,
            last_scraping_num_articles INTEGER,
            last_scraping_num_search_terms INTEGER,
            last_scraping_status TEXT
        )
        """

        try:
            conn.execute(create_articles_table)
            conn.execute(create_search_terms_table)
            conn.execute(create_metadata_table)
            logging.info(f"Database initialized. Mode: {self.db_mode}")
        except Exception as e:
            logging.error(f"Failed to create tables: {e}")
        finally:
            conn.close()

    def list_terms(self):
        conn = self._connect()
        if not conn:
            return []
        try:
            cursor = conn.execute("SELECT Begriffe FROM search_terms")
            terms = [row[0] for row in cursor.fetchall()]
            logging.info(f"Fetched {len(terms)} search terms.")
        except Exception as e:
            logging.error(f"Error fetching search terms: {e}")
            terms = []
        conn.close()
        return terms

    def insert_terms(self, terms):
        conn = self._connect()
        if not conn:
            return
        for term in terms:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO search_terms (Begriffe) VALUES (?)", (term,)
                )
            except Exception as e:
                logging.error(f"Error inserting search term {term}: {e}")
        conn.close()

    def delete_term(self, term):
        conn = self._connect()
        if not conn:
            return
        try:
            conn.execute("DELETE FROM articles WHERE begriff=?", (term,))
            conn.execute("DELETE FROM search_terms WHERE Begriffe=?", (term,))
            logging.info(f"Deleted search term and associated articles: {term}")
        except Exception as e:
            logging.error(f"Error deleting term {term}: {e}")
        conn.close()

    def insert_articles(self, df):
        """
        Inserts articles from a DataFrame into the SQLite database.

        Args:
            df (pd.DataFrame): DataFrame containing articles with columns 'name', 'begriff', 'date', 'url'.

        Returns:
            tuple: (number of new records inserted, number of duplicates ignored)
        """
        conn = self._connect()
        if not conn:
            return 0, 0

        inserted_count = 0
        duplicate_count = 0

        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Improves concurrency

            # Record total changes before insertion
            before_changes = conn.total_changes

            # Prepare data for insertion with 'no' field
            data_to_insert = []
            for _, row in df.iterrows():
                name = row.get("name")
                begriff = row.get("begriff")
                date = row.get("date")
                url = row.get("url")

                # Generate hash based on 'url' and 'begriff'
                hash_input = f"{url}{begriff}"
                no_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

                data_to_insert.append((name, begriff, date, url, no_hash))

            # Perform bulk insert using executemany
            conn.executemany(
                """
                INSERT OR IGNORE INTO articles (name, begriff, date, url, no) 
                VALUES (?, ?, ?, ?, ?)
                """,
                data_to_insert,
            )

            # Record total changes after insertion
            after_changes = conn.total_changes

            # Calculate number of inserted rows
            inserted_count = after_changes - before_changes

            # Calculate duplicates ignored
            total_attempted = len(data_to_insert)
            duplicate_count = total_attempted - inserted_count

            # Commit the transaction
            conn.commit()

            # Log the results
            logging.info(f"Inserted {inserted_count} new records into the database.")
            logging.info(f"Ignored {duplicate_count} duplicate records.")

        except sqlite3.Error as e:
            logging.error(f"DB Insert Error: {e}")
        finally:
            conn.close()

        return inserted_count

    def get_articles(self, search_query=None):
        conn = self._connect()
        if not conn:
            return []
        try:
            if search_query:
                like_query = f"%{search_query}%"
                cursor = conn.execute(
                    """
                    SELECT begriff, date, name, url 
                    FROM articles 
                    WHERE begriff LIKE ? OR name LIKE ? OR url LIKE ? 
                    ORDER BY date DESC
                    """,
                    (like_query, like_query, like_query),
                )
            else:
                cursor = conn.execute(
                    "SELECT begriff, date, name, url FROM articles ORDER BY date DESC"
                )
            rows = cursor.fetchall()
            articles = [list(row) for row in rows]
            logging.info(f"Fetched {len(articles)} articles from the database.")
        except Exception as e:
            logging.error(f"Error fetching articles: {e}")
            articles = []
        conn.close()
        return articles

    def insert_metadata(self, timestamp, inserted_articles, num_search_terms, status):
        conn = self._connect()
        if not conn:
            return
        try:
            conn.execute(
                """
                INSERT INTO metadata (last_scraping_timestamp, last_scraping_num_articles, 
                                      last_scraping_num_search_terms, last_scraping_status)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, inserted_articles, num_search_terms, status),
            )
            logging.info("Metadata updated successfully.")
        except Exception as e:
            logging.error(f"Error updating metadata: {e}")
        conn.close()

    def get_latest_metadata(self):
        conn = self._connect()
        if not conn:
            return None
        try:
            cursor = conn.execute("""
                SELECT last_scraping_timestamp, last_scraping_num_articles, 
                       last_scraping_num_search_terms, last_scraping_status
                FROM metadata
                ORDER BY id DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                metadata = {
                    "last_scraping_timestamp": row[0],
                    "last_scraping_status": row[3],
                }
                logging.info("Fetched latest metadata.")
                return metadata
            else:
                return None
        except Exception as e:
            logging.error(f"Error fetching metadata: {e}")
            return None
        finally:
            conn.close()

    def count_articles(self):
        conn = self._connect()
        if not conn:
            return 0
        count = 0
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM articles")
            row = cursor.fetchone()
            if row:
                count = row[0]
        except Exception as e:
            logging.error(f"Error counting articles: {e}")
        conn.close()
        return count
