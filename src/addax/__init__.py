# src/addax/__init__.py
"""Addax: A Text Analysis Data Library."""

# read version from installed package
from importlib.metadata import version

__version__ = version("addax")


from addax.addax import (
    analyze_sentiment_dataframe,
    analyze_sentiment_text,
    label_polarity,
    label_subjectivity,
    logger,
    normalize_column_name,
    normalize_series_text,
    process_sentiment,
    read_csv,
    remove_rows_missing_target,
    standardize_headers,
    standardize_target_col_data,
    standardize_target_col_name,
)

__all__ = [
    "__version__",
    "analyze_sentiment_dataframe",
    "analyze_sentiment_text",
    "label_polarity",
    "label_subjectivity",
    "logger",
    "normalize_column_name",
    "normalize_series_text",
    "process_sentiment",
    "read_csv",
    "remove_rows_missing_target",
    "standardize_headers",
    "standardize_target_col_data",
    "standardize_target_col_name",
]
