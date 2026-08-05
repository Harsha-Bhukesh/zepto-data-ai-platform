# Module 1 — Data Pipeline

## Overview

This module implements an end-to-end **data engineering pipeline** for catalog-style book data.

The pipeline demonstrates how raw web data can be collected, cleaned, transformed, stored in a normalized relational database, queried using SQL, and analyzed using pandas.

### Pipeline Workflow

```text
Public Web Source
       ↓
Web Scraping
       ↓
Data Cleaning & Validation
       ↓
GBP → INR Conversion
       ↓
Normalized SQLite Database
       ↓
SQL Queries
       ↓
Pandas Analysis
       ↓
SQL JOIN vs Pandas merge Validation
```

The module uses **Books to Scrape**, a public website specifically designed for practicing web scraping.

No login, API key, or paid service is required.

---

## Data Source

The book data is scraped from:

**Books to Scrape:** [Books to Scrape](http://books.toscrape.com/?utm_source=chatgpt.com)

The website is used only as the source for the scraping exercise. The data is collected programmatically using `requests` and `BeautifulSoup`.

### Categories Used

The pipeline collects books from the following three categories:

* **Travel:** [Travel category](https://books.toscrape.com/catalogue/category/books/travel_2/index.html?utm_source=chatgpt.com)
* **Mystery:** [Mystery category](https://books.toscrape.com/catalogue/category/books/mystery_3/index.html?utm_source=chatgpt.com)
* **Historical Fiction:** [Historical Fiction category](https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html?utm_source=chatgpt.com)

### Dataset Summary

| Category           |  Books |
| ------------------ | -----: |
| Travel             |     11 |
| Mystery            |     32 |
| Historical Fiction |     26 |
| **Total**          | **69** |

The final dataset contains **69 books across 3 categories**, satisfying the requirement of at least 60 books across at least 3 categories.

---

# Project Structure

```text
data_pipeline/
│
├── README.md
├── requirements.txt
├── scraper.py
├── database.py
├── queries.py
└── books.db
```

### File Description

| File               | Description                                        |
| ------------------ | -------------------------------------------------- |
| `scraper.py`       | Scrapes, cleans, converts, and validates book data |
| `database.py`      | Creates and populates the SQLite database          |
| `queries.py`       | Executes SQL queries and performs pandas analysis  |
| `requirements.txt` | Contains required Python packages                  |
| `books.db`         | SQLite database containing the processed data      |
| `README.md`        | Module documentation                               |

---

# 1. Web Scraping

The scraping component uses:

* `requests`
* `BeautifulSoup`

The scraper sends HTTP requests to the selected category pages and extracts book information from the HTML.

The following fields are collected:

```text
title
price
star_rating
availability
category
```

The scraping process is completely automated and does not require manual copy-pasting.

---

# 2. Data Cleaning

The raw scraped values are cleaned and converted into appropriate data types before being stored in the database.

## Price Cleaning

The website provides prices with the GBP currency symbol.

Example:

```text
£45.17
```

The currency symbol is removed and the value is converted to a floating-point number:

```text
45.17
```

The resulting column is:

```text
price_gbp
```

with a floating-point data type.

---

## Star Rating Conversion

The website provides ratings as text:

```text
One
Two
Three
Four
Five
```

These are converted to integers:

```text
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

The resulting column is:

```text
rating
```

---

## Availability Conversion

Availability is provided as text, for example:

```text
In stock
```

This is converted into a boolean value:

```text
In stock → True
```

When stored in SQLite, the boolean value is represented as:

```text
True  → 1
False → 0
```

The resulting column is:

```text
in_stock
```

---

## Handling Parsing Errors

The pipeline includes validation for scraped fields so that unexpected values do not cause the entire pipeline to fail.

For numeric fields, invalid values can be handled using median imputation when appropriate.

If a row contains data that cannot be safely interpreted and cannot reasonably be imputed, the row can be removed.

For the final scraped dataset used in this submission, all required fields were successfully parsed and no rows needed to be removed.

---

# 3. Currency Conversion

The project requires a fixed baseline conversion rate:

```text
1 GBP = 105.50 INR
```

This is an **artificial project-defined constant**, not a live exchange rate.

No external currency API is used.

The conversion is calculated as:

```text
price_inr = price_gbp × 105.50
```

### Example

```text
45.17 GBP × 105.50 = 4765.435 INR
```

The resulting column is:

```text
price_inr
```

Using the fixed rate makes the pipeline fully reproducible and follows the required project baseline.

---

# 4. Database Design

The cleaned data is stored in a normalized **SQLite relational database**.

The database contains two main tables:

```text
categories
books
```

## Entity Relationship

```text
┌─────────────────────┐
│     categories      │
├─────────────────────┤
│ category_id (PK)    │
│ category_name       │
└──────────┬──────────┘
           │
           │ 1-to-many
           │
           ▼
┌─────────────────────┐
│       books         │
├─────────────────────┤
│ book_id (PK)        │
│ title               │
│ price_gbp           │
│ price_inr           │
│ rating              │
│ in_stock            │
│ category_id (FK)    │
└─────────────────────┘
```

---

## Categories Table

```text
categories
-----------------------------
category_id    INTEGER PRIMARY KEY
category_name  TEXT UNIQUE
```

Example:

```text
1 → Travel
2 → Mystery
3 → Historical Fiction
```

---

## Books Table

```text
books
------------------------------------------------
book_id       INTEGER PRIMARY KEY
title         TEXT
price_gbp     REAL
price_inr     REAL
rating        INTEGER
in_stock      INTEGER
category_id   INTEGER FOREIGN KEY
```

The `category_id` column references `categories.category_id`.

This normalized structure avoids repeatedly storing category names for every book and establishes a proper primary-key/foreign-key relationship.

---

# 5. Database Population

`database.py` runs the database creation process.

The process is:

```text
Scrape books
     ↓
Clean data
     ↓
Validate data
     ↓
Create SQLite tables
     ↓
Insert categories
     ↓
Insert books
     ↓
Verify database
```

### Database Verification

The final database contains:

```text
Categories: 3
Books: 69
```

Books by category:

```text
Historical Fiction: 26
Mystery: 32
Travel: 11
```

---

# 6. SQL Queries

The project contains six SQL queries in `queries.py`.

Together, these queries demonstrate all the SQL operations required by the assignment.

---

## Query 1 — SELECT + WHERE

Find books with a rating of at least 4.

```sql
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
ORDER BY rating DESC;
```

Result:

```text
27 books
```

Demonstrates:

* `SELECT`
* `WHERE`
* `ORDER BY`

---

## Query 2 — ORDER BY + LIMIT

Find the 10 most expensive books.

```sql
SELECT title, price_gbp, rating
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
```

Demonstrates:

* `ORDER BY`
* `LIMIT`

---

## Query 3 — DISTINCT

Retrieve the unique book categories.

```sql
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
```

Result:

```text
Historical Fiction
Mystery
Travel
```

Demonstrates:

* `DISTINCT`

---

## Query 4 — BETWEEN

Find books priced between £20 and £40.

```sql
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp ASC;
```

Demonstrates:

* `BETWEEN`
* `WHERE`
* `ORDER BY`

---

## Query 5 — IN + JOIN

Find books belonging to the Travel or Mystery categories.

```sql
SELECT
    b.title,
    b.price_gbp,
    b.rating,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
WHERE c.category_name IN ('Travel', 'Mystery')
ORDER BY b.rating DESC, b.price_gbp DESC;
```

Demonstrates:

* `JOIN`
* `IN`
* `WHERE`
* `ORDER BY`

---

## Query 6 — JOIN

Find the 10 highest-rated books along with their categories.

```sql
SELECT
    b.book_id,
    b.title,
    b.price_gbp,
    b.rating,
    b.in_stock,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.price_gbp DESC
LIMIT 10;
```

This query demonstrates the required relationship between the `books` and `categories` tables.

---

# 7. Pandas Analysis

SQL results are also loaded into pandas DataFrames using:

```python
pd.read_sql()
```

The project reads multiple query results into pandas, including:

* Query 1
* Query 2
* JOIN query

This demonstrates integration between a relational database and a Python data-analysis workflow.

---

# 8. Pandas JOIN Reproduction

The SQL JOIN is independently reproduced using:

```python
pd.merge()
```

The two database tables are first loaded into pandas:

```text
books
categories
```

They are then merged using the common `category_id` column:

```python
df_merge = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)
```

The result is sorted and limited using the same logic as the SQL JOIN query.

---

# 9. SQL JOIN vs Pandas merge Validation

The SQL JOIN result and the pandas merge result are compared using:

```python
df_sql_join.equals(df_merge)
```

The final validation produced:

```text
Do SQL JOIN and pandas merge produce equivalent results? True
```

Therefore, the pandas implementation successfully reproduces the SQL JOIN result.

---

# 10. Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

### Main Dependencies

```text
requests
beautifulsoup4
pandas
```

SQLite is provided through Python's standard library, so no separate SQLite installation is required.

---

# 11. Running the Pipeline

## Step 1 — Create the Database

From inside the `data_pipeline` directory, run:

```bash
python database.py
```

This will scrape the data, clean it, create the database, insert the records, and verify the results.

---

## Step 2 — Run SQL and Pandas Analysis

After the database has been created, run:

```bash
python queries.py
```

This executes all six SQL queries and performs the pandas analysis.

The final validation should show:

```text
Do SQL JOIN and pandas merge produce equivalent results? True
```

---

# 12. Reproducibility

The complete pipeline can be regenerated from the source website.

The database does not depend on any paid service or external API.

The required currency conversion uses the fixed project-defined rate:

```text
1 GBP = 105.50 INR
```

Therefore, the pipeline can be reproduced without API keys or paid services.

---

# 13. Design Decisions

### Web Scraping

`requests` and `BeautifulSoup` were selected because they are lightweight, widely used Python tools for retrieving and parsing static HTML content.

### Data Cleaning

Raw website values are converted into suitable numerical, categorical, and boolean representations before database insertion.

### Currency Conversion

The required fixed conversion rate of **1 GBP = 105.50 INR** is used instead of a live exchange-rate API. This keeps the result deterministic and follows the assignment requirements.

### Database

SQLite was selected because it is lightweight, serverless, reproducible, and integrates directly with Python.

### Normalization

The category information is separated into a `categories` table and referenced by `category_id` from the `books` table. This avoids redundant category names and establishes a proper relational structure.

### SQL Analysis

SQL is used for filtering, sorting, limiting, selecting distinct values, range filtering, category filtering, and relational JOIN operations.

### Pandas Analysis

Pandas is used to load SQL results and independently reproduce the JOIN operation using `pd.merge()`.

---

# 14. Module Completion Summary

The Data Pipeline module implements the complete required workflow:

```text
✓ Web scraping
✓ requests
✓ BeautifulSoup
✓ 69 books
✓ 3 categories
✓ Data cleaning
✓ price_gbp conversion
✓ rating conversion
✓ in_stock conversion
✓ Fixed GBP → INR conversion
✓ Normalized SQLite database
✓ Primary key / Foreign key relationship
✓ SELECT / WHERE
✓ ORDER BY
✓ LIMIT
✓ DISTINCT
✓ BETWEEN
✓ IN
✓ SQL JOIN
✓ pd.read_sql()
✓ pd.merge()
✓ SQL vs Pandas validation
✓ End-to-end reproducible pipeline
```

### Final Result

```text
Dataset: 69 books
Categories: 3
Database: SQLite
Currency Rate: 1 GBP = 105.50 INR
SQL Queries: 6
JOIN Validation: True
```

The module demonstrates a complete raw-to-relational data pipeline:

```text
SCRAPE
   ↓
CLEAN
   ↓
TRANSFORM
   ↓
STORE
   ↓
QUERY
   ↓
ANALYZE
   ↓
VALIDATE
```

