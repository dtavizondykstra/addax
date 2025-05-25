# src/addax/__init__.py
"""Addax: A DataFrame Processing Library"""
# read version from installed package
from importlib.metadata import version

__version__ = version("addax")


from addax.addax import (
    read_csv,
    standardize_headers,
    format_text_data_target,
    remove_rows_missing_target,
    analyze_sentiment_text,
    analyze_sentiment_dataframe,
    process_text_column,
)

__all__ = [
    "read_csv",
    "standardize_headers",
    "format_text_data_target",
    "remove_rows_missing_target",
    "analyze_sentiment_text",
    "analyze_sentiment_dataframe",
    "process_text_column",
]
