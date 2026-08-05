import sqlite3
import pandas as pd


# Database configuration
DB_NAME = "books.db"


def run_sql_queries(connection):
    """Execute the required SQL queries and display their results."""

    queries = {
        "query_1": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4
            ORDER BY rating DESC
        """,

        "query_2": """
            SELECT title, price_gbp, rating
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10
        """,

        "query_3": """
            SELECT DISTINCT category_name
            FROM categories
            ORDER BY category_name
        """,

        "query_4": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE price_gbp BETWEEN 20 AND 40
            ORDER BY price_gbp ASC
        """,

        "query_5": """
            SELECT
                b.title,
                b.price_gbp,
                b.rating,
                c.category_name
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            WHERE c.category_name IN ('Travel', 'Mystery')
            ORDER BY b.rating DESC, b.price_gbp DESC
        """,

        "query_6": """
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
            LIMIT 10
        """
    }

    results = {}

    for query_name, query in queries.items():

        result = connection.execute(query).fetchall()

        results[query_name] = result

        print("\n" + "=" * 80)
        print(query_name.upper())
        print("=" * 80)

        print(query.strip())

        print("\nOutput:")

        for row in result:
            print(row)

        print(f"\nRows returned: {len(result)}")

    return queries, results


def run_pandas_analysis(connection):
    """Read SQL results using pandas and reproduce the JOIN with pd.merge()."""

    # ---------------------------------------------------------
    # Query 1 using pd.read_sql()
    # ---------------------------------------------------------

    query_1 = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC
    """

    df_query_1 = pd.read_sql(query_1, connection)

    print("\n" + "=" * 80)
    print("PANDAS RESULT - QUERY 1")
    print("=" * 80)

    print(df_query_1.head(10))


    # ---------------------------------------------------------
    # Query 2 using pd.read_sql()
    # ---------------------------------------------------------

    query_2 = """
        SELECT title, price_gbp, rating
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
    """

    df_query_2 = pd.read_sql(query_2, connection)

    print("\n" + "=" * 80)
    print("PANDAS RESULT - QUERY 2")
    print("=" * 80)

    print(df_query_2)


    # ---------------------------------------------------------
    # SQL JOIN using pd.read_sql()
    # ---------------------------------------------------------

    join_query = """
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
        LIMIT 10
    """

    df_sql_join = pd.read_sql(join_query, connection)

    print("\n" + "=" * 80)
    print("SQL JOIN RESULT USING pd.read_sql()")
    print("=" * 80)

    print(df_sql_join)


    # ---------------------------------------------------------
    # Load individual tables into pandas
    # ---------------------------------------------------------

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )


    # ---------------------------------------------------------
    # Reproduce SQL JOIN using pd.merge()
    # ---------------------------------------------------------

    df_merge = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    # Apply the same ordering and LIMIT as the SQL JOIN
    df_merge = (
        df_merge
        .sort_values(
            by=["rating", "price_gbp"],
            ascending=[False, False]
        )
        .head(10)
        .reset_index(drop=True)
    )

    # Match the SQL column order
    df_merge = df_merge[
        [
            "book_id",
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "category_name"
        ]
    ]


    # Reset SQL result index for comparison
    df_sql_join = df_sql_join.reset_index(drop=True)


    # ---------------------------------------------------------
    # Compare SQL JOIN and pandas merge
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("PANDAS MERGE RESULT")
    print("=" * 80)

    print(df_merge)


    equivalent = df_sql_join.equals(df_merge)

    print("\n" + "=" * 80)
    print("SQL JOIN vs pandas merge")
    print("=" * 80)

    print(
        "Do SQL JOIN and pandas merge produce equivalent results?",
        equivalent
    )

    return (
        df_query_1,
        df_query_2,
        df_sql_join,
        df_merge,
        equivalent
    )


def main():
    """Run all SQL queries and pandas analysis."""

    print("Starting SQL and Pandas analysis...")
    print("=" * 80)

    connection = sqlite3.connect(DB_NAME)

    try:

        # Execute required SQL queries
        queries, results = run_sql_queries(connection)

        # Perform pandas analysis
        (
            df_query_1,
            df_query_2,
            df_sql_join,
            df_merge,
            equivalent
        ) = run_pandas_analysis(connection)

        # Final validation
        if equivalent:
            print("\n" + "=" * 80)
            print("VALIDATION SUCCESSFUL")
            print("=" * 80)
            print("SQL JOIN and pandas merge results are equivalent.")
        else:
            print("\n" + "=" * 80)
            print("VALIDATION FAILED")
            print("=" * 80)
            print("SQL JOIN and pandas merge results do not match.")

    finally:
        connection.close()

    print("\nSQL and Pandas analysis completed.")


if __name__ == "__main__":
    main()
