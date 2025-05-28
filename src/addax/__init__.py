# src/addax/__init__.py
"""Addax: A DataFrame Processing Library"""
# read version from installed package
from importlib.metadata import version

__version__ = version("addax")


from addax.addax import (
    read_csv,
    normalize_column_name,
    normalize_series_text,
    standardize_target_col_name,
    standardize_headers,
    standardize_target_col_data,
    remove_rows_missing_target,
    analyze_sentiment_text,
    analyze_sentiment_dataframe,
    process_sentiment,
    label_polarity,
    label_subjectivity,
    logger,
)

__all__ = [
    "read_csv",
    "normalize_column_name",
    "normalize_series_text",
    "standardize_target_col_name",
    "standardize_headers",
    "standardize_target_col_data",
    "remove_rows_missing_target",
    "analyze_sentiment_text",
    "analyze_sentiment_dataframe",
    "process_sentiment",
    "label_polarity",
    "label_subjectivity",
    "logger",
    "__version__",
]
