import pandas as pd

from src.cleaning.clean_data import REQUIRED_COLUMNS, clean_transactions


def _transactions():
    rows = [
        ("A001", "S1", None, 2, "2011-01-01", 10.0, 10001.0, "United Kingdom"),
        ("A001", "S1", None, 2, "2011-01-01", 10.0, 10001.0, "United Kingdom"),
        ("A002", "POST", "Postage", 1, "2011-01-02", 2.0, 10002.0, "United Kingdom"),
        ("A003", "S2", "Free item", 1, "2011-01-03", 0.0, 10003.0, "Germany"),
        ("A004", "S3", "Returned units", -3, "2011-01-04", 5.0, 10004.0, "France"),
        ("C005", "S4", "Cancellation", 1, "2011-01-05", 4.0, 10005.0, "France"),
        ("A006", "S5", "Guest", 1, "2011-01-06", 3.0, None, "Germany"),
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_drops_missing_customer_id():
    cleaned = clean_transactions(_transactions())
    assert "A006" not in set(cleaned["Invoice"])


def test_drops_exact_duplicates():
    cleaned = clean_transactions(_transactions())
    assert cleaned["Invoice"].value_counts()["A001"] == 1


def test_drops_cancellation_invoices():
    cleaned = clean_transactions(_transactions())
    assert "C005" not in set(cleaned["Invoice"])


def test_fills_missing_description():
    cleaned = clean_transactions(_transactions())
    assert cleaned.loc[cleaned["Invoice"] == "A001", "Description"].item() == (
        "Unknown Product"
    )


def test_output_dtypes():
    cleaned = clean_transactions(_transactions())
    assert pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"])
    assert pd.api.types.is_integer_dtype(cleaned["Customer ID"])


def test_default_flags_keep_non_product_and_zero_price():
    cleaned = clean_transactions(_transactions())
    assert "A002" in set(cleaned["Invoice"])
    assert "A003" in set(cleaned["Invoice"])


def test_true_flags_drop_non_product_and_nonpositive_price():
    cleaned = clean_transactions(
        _transactions(), exclude_non_product=True, drop_nonpositive_price=True
    )
    assert set(cleaned["Invoice"]) == {"A001", "A004"}


def test_preserves_negative_quantity_for_non_cancellations():
    cleaned = clean_transactions(_transactions())
    # Documented behavior: non-cancellation negative quantities survive cleaning.
    # See docs/known_issues.md.
    assert cleaned.loc[cleaned["Invoice"] == "A004", "Quantity"].item() == -3