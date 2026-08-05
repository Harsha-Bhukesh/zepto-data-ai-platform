import sqlite3
import pandas as pd

from scraper import scrape_books, clean_data, validate_data


# Database configuration
DB_NAME = "books.db"


def create_connection():
    """Create and return a SQLite database connection."""
    return sqlite3.connect(DB_NAME)


def create_tables(connection):
    """Create the normalized database schema."""

    cursor = connection.cursor()

    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON")

    # Categories table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    # Books table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    connection.commit()

    print("Database tables created successfully.")


def insert_categories(connection, books_df):
    """Insert unique categories into the categories table."""

    cursor = connection.cursor()

    categories = books_df["category"].dropna().unique()

    for category in categories:
        cursor.execute(
            """
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    connection.commit()

    print(f"Inserted {len(categories)} categories.")


def insert_books(connection, books_df):
    """Insert cleaned books and their category foreign keys."""

    cursor = connection.cursor()

    # Create category name → category ID mapping
    cursor.execute("""
        SELECT category_id, category_name
        FROM categories
    """)

    category_map = {
        category_name: category_id
        for category_id, category_name in cursor.fetchall()
    }

    for _, row in books_df.iterrows():

        category_id = category_map[row["category"]]

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_id
            )
        )

    connection.commit()

    print(f"Inserted {len(books_df)} books.")


def verify_database(connection):
    """Verify the number of categories and books."""

    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM categories")
    category_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM books")
    book_count = cursor.fetchone()[0]

    print("\nDatabase verification")
    print("-" * 30)
    print(f"Categories: {category_count}")
    print(f"Books: {book_count}")

    # Verify books by category
    cursor.execute("""
        SELECT
            c.category_name,
            COUNT(b.book_id) AS book_count
        FROM categories c
        JOIN books b
            ON c.category_id = b.category_id
        GROUP BY c.category_name
        ORDER BY c.category_name
    """)

    results = cursor.fetchall()

    print("\nBooks by category:")

    for category, count in results:
        print(f"{category}: {count}")

    return category_count, book_count


def build_database():
    """Run the complete data pipeline and build the SQLite database."""

    print("Starting database pipeline...")
    print("=" * 50)

    # Step 1: Scrape raw books
    print("\nStep 1: Scraping books")
    books_df = scrape_books()

    # Step 2: Clean and transform data
    print("\nStep 2: Cleaning data")
    books_df = clean_data(books_df)

    # Step 3: Validate data
    print("\nStep 3: Validating data")
    validate_data(books_df)

    # Step 4: Create database
    print("\nStep 4: Creating SQLite database")
    connection = create_connection()

    try:
        # Step 5: Create tables
        create_tables(connection)

        # Step 6: Insert categories
        insert_categories(connection, books_df)

        # Step 7: Insert books
        insert_books(connection, books_df)

        # Step 8: Verify database
        verify_database(connection)

    finally:
        connection.close()

    print("\n" + "=" * 50)
    print("Database pipeline completed successfully!")
    print(f"Database file: {DB_NAME}")


if __name__ == "__main__":
    build_database()
