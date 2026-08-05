import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# Project configuration
BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

CATEGORIES = {
    "Travel": "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": (
        "https://books.toscrape.com/catalogue/category/books/"
        "historical-fiction_4/index.html"
    ),
}


def fetch_page(url):
    """Fetch a webpage and return a BeautifulSoup object."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def scrape_category(category_name, category_url):
    """Scrape all books from a category, including pagination."""

    books_data = []
    current_url = category_url

    while current_url:

        soup = fetch_page(current_url)

        books = soup.select("article.product_pod")

        for book in books:

            title = book.h3.a.get("title", "").strip()

            price = book.select_one(".price_color").get_text(strip=True)

            star_element = book.select_one("p.star-rating")
            star_rating = (
                star_element.get("class", ["", ""])[1]
                if star_element
                else ""
            )

            availability_element = book.select_one(".availability")
            availability = (
                availability_element.get_text(" ", strip=True)
                if availability_element
                else ""
            )

            books_data.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category_name,
                }
            )

        # Handle pagination automatically
        next_button = soup.select_one("li.next a")

        if next_button:
            current_url = urljoin(current_url, next_button["href"])
        else:
            current_url = None

        # Small delay between requests
        time.sleep(0.2)

    return books_data


def scrape_books():
    """Scrape books from all configured categories."""

    all_books = []

    for category_name, category_url in CATEGORIES.items():

        category_books = scrape_category(
            category_name,
            category_url
        )

        all_books.extend(category_books)

        print(
            f"{category_name}: "
            f"{len(category_books)} books scraped"
        )

    return pd.DataFrame(all_books)


def clean_data(df):
    """Clean scraped fields and create required derived columns."""

    df = df.copy()

    # Convert price from GBP text to numeric
    df["price_gbp"] = (
        df["price"]
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # Median imputation for invalid numeric prices
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # Convert star ratings from text to integers
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    df["rating"] = df["star_rating"].map(rating_map)

    # Drop rows where rating could not be parsed
    df = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].astype(int)

    # Convert availability text to boolean
    df["in_stock"] = (
        df["availability"]
        .str.strip()
        .str.lower()
        .eq("in stock")
    )

    # Required fixed-rate currency conversion
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df


def validate_data(df):
    """Validate the final dataset against project requirements."""

    required_columns = [
        "title",
        "price",
        "star_rating",
        "availability",
        "category",
        "price_gbp",
        "rating",
        "in_stock",
        "price_inr",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} books found. "
            "At least 60 books are required."
        )

    if df["category"].nunique() < 3:
        raise ValueError(
            "At least 3 different categories are required."
        )

    if df["price_gbp"].isna().any():
        raise ValueError("Missing values found in price_gbp.")

    if df["rating"].isna().any():
        raise ValueError("Missing values found in rating.")

    if df["price_inr"].isna().any():
        raise ValueError("Missing values found in price_inr.")

    if not df["rating"].between(1, 5).all():
        raise ValueError("Rating must be between 1 and 5.")

    print("\nData validation successful!")
    print(f"Total books: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Currency rate: 1 GBP = {GBP_TO_INR} INR")


def main():
    """Run the complete scraping and cleaning pipeline."""

    print("Starting Zepto Data Pipeline...")
    print("-" * 40)

    # Scrape raw data
    books_df = scrape_books()

    print("-" * 40)
    print(f"Raw books scraped: {len(books_df)}")

    # Clean and transform data
    books_df = clean_data(books_df)

    # Validate final dataset
    validate_data(books_df)

    # Display sample
    print("\nCleaned dataset:")
    print(
        books_df[
            [
                "title",
                "price_gbp",
                "rating",
                "in_stock",
                "price_inr",
                "category",
            ]
        ].head()
    )

    print("\nFinal data types:")
    print(books_df.dtypes)

    return books_df


if __name__ == "__main__":
    books_df = main()
