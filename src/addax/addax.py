# addax.py
"""Addax: A Text Analysis Library."""

import logging
import re

import pandas as pd
from textblob import TextBlob

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def process_sentiment(
    df: pd.DataFrame,
    target_column: str,
    include_subjectivity: bool = True,
    label: bool = True,
) -> pd.DataFrame:
    """End-to-end pipeline that standardizes headers, formats the target column,
    removes missing entries, and then runs sentiment analysis with optional labeling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column to process.
        include_subjectivity (bool): Whether to include subjectivity score.
        label (bool): Whether to add categorical labels.

    Returns:
        pd.DataFrame: Processed DataFrame with 'polarity', optional 'subjectivity', and optional labels.

    """
    df_proc = standardize_headers(df)
    target_col = standardize_target_col_name(target_column)
    # Ensure target column exists
    if target_col not in df_proc.columns:
        logger.warning(
            f"Target column '{target_column}' not found in standardized headers. No processing applied.",
        )
        return df_proc.copy()
    logger.info(f"Processing text column '{target_col}' for sentiment analysis.")

    df_proc = standardize_target_col_data(df_proc, target_column)
    df_proc = analyze_sentiment_dataframe(
        df_proc,
        target_column,
        include_subjectivity=include_subjectivity,
        label=label,
    )
    return df_proc


def read_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame of the CSV data, or empty DataFrame on error.

    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read CSV: {file_path}")
        return df
    except Exception as e:
        logger.exception(f"Error reading CSV file '{file_path}': {e}")
        return pd.DataFrame()


def standardize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame column headers by normalizing each column name:
    lowercases, replaces spaces with underscores, and removes non-alphanumeric
    and non-underscore characters.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standardized column names (lowercase, underscores
                     instead of spaces, and only alphanumeric/underscore characters).

    """
    df_std = df.copy()
    df_std.columns = df_std.columns.map(normalize_column_name)
    logger.info("Standardized headers to lowercase and underscores.")
    return df_std


def standardize_target_col_name(col_name: str) -> str:
    """Normalize a target column name to match the standardized format used by standardize_headers().

    Converts to lowercase, replaces spaces with underscores, and removes non-alphanumeric characters.

    Args:
        col_name (str): The target column name to normalize.

    Returns:
        str: The normalized column name with lowercase letters, underscores
             instead of spaces, and only alphanumeric/underscore characters.

    """
    return normalize_column_name(col_name)


def normalize_column_name(col: str) -> str:
    """Normalize a column name by lowercasing, replacing spaces with underscores,
    and removing non-alphanumeric/underscore characters.

    Args:
        col (str): The column name to normalize.

    Returns:
        str: The normalized column name with lowercase letters, underscores
             instead of spaces, and only alphanumeric/underscore characters.

    """
    col = col.lower()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^0-9a-zA-Z_]+", "", col)
    return col


def standardize_target_col_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Clean textual data in the specified column by lowercasing, and removing special characters,
    and removing rows with missing or empty values.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        target_column (str): Name of the column to format.

    Returns:
        pd.DataFrame: DataFrame with rows containing missing target values removed
                     and cleaned text in the target column (lowercased with
                     special characters removed).

    """
    target_column = standardize_target_col_name(target_column)
    if target_column not in df.columns:
        logger.warning(
            f"Target column '{target_column}' not found. No formatting applied.",
        )
        return df.copy()

    logger.info(f"Formatting text data in target column '{target_column}'.")
    df_clean = df.copy()

    # Remove rows where target column is missing or empty
    df_clean = remove_rows_missing_target(df_clean, target_column)
    logger.info(
        f"Formatted text in column '{target_column}': lowercased and removed special characters.",
    )

    # Format the target column: lowercasing and removing special characters
    df_clean[target_column] = normalize_series_text(df_clean[target_column])

    # Remove rows where target column is missing or empty
    df_clean = remove_rows_missing_target(df_clean, target_column)
    logger.info(
        f"Formatted text in column '{target_column}': lowercased and removed special characters.",
    )
    return df_clean


def remove_rows_missing_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Remove rows where the target column has missing or empty values.

    Standardizes the target column name and filters out rows containing NaN values,
    empty strings, or whitespace-only strings in the specified column.

    Args:
        df (pd.DataFrame): Input DataFrame to filter.
        target_column (str): Name of the column to check for missing/empty values.
                           Will be standardized to lowercase with underscores.

    Returns:
        pd.DataFrame: Filtered DataFrame with rows containing missing, empty, or
                     whitespace-only values in the target column removed. Returns
                     a copy of the original DataFrame if target column not found.

    """
    target_column = standardize_target_col_name(target_column)

    if target_column not in df.columns:
        logger.warning(
            f"Target column '{target_column}' not in DataFrame. No rows removed.",
        )
        return df.copy()

    # First drop NaNs, then drop empty or all whitespace strings
    mask = df[target_column].notna() & df[target_column].astype(  # keep non-null
        str,
    ).str.strip().ne(
        "",
    )  # drop "" or "   "
    df_filtered = df[mask].copy()
    removed = len(df) - len(df_filtered)
    logger.info(f"Removed {removed} rows with missing or empty '{target_column}'.")
    return df_filtered


def normalize_series_text(series: pd.Series) -> pd.Series:
    """Normalize text data in a pandas Series by converting all values to strings,
    lowercasing them, and removing any character that is not a Unicode letter
    or whitespace. This preserves accented letters and non-Latin scripts.

    Args:
        series (pd.Series): Input pandas Series containing text data to normalize.

    Returns:
        pd.Series: Normalized Series with only lowercase alphabetic (Unicode)
                   characters and spaces.

    """

    def _clean(txt: str) -> str:
        txt = str(txt).lower()
        return "".join(ch for ch in txt if ch.isalpha() or ch.isspace())

    return series.astype(str).map(_clean)


def analyze_sentiment_dataframe(
    df: pd.DataFrame,
    target_column: str,
    include_subjectivity: bool = True,
    label: bool = True,
) -> pd.DataFrame:
    """Apply sentiment analysis over a DataFrame column of text, with optional labels.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the column containing text.
        include_subjectivity (bool): Whether to include subjectivity score.
        label (bool): Whether to add categorical labels for polarity and subjectivity.

    Returns:
        pd.DataFrame: DataFrame with 'polarity', optional 'subjectivity', and optional 'polarity_label' and 'subjectivity_label'.

    """
    target_column = standardize_target_col_name(target_column)
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame.")

    df_out = df.copy()
    df_out["polarity"] = df_out[target_column].fillna("").apply(lambda t: analyze_sentiment_text(t)["polarity"])

    if include_subjectivity:
        df_out["subjectivity"] = (
            df_out[target_column].fillna("").apply(lambda t: analyze_sentiment_text(t)["subjectivity"])
        )

    if label:
        df_out["polarity_label"] = df_out["polarity"].apply(label_polarity)
        if include_subjectivity:
            df_out["subjectivity_label"] = df_out["subjectivity"].apply(
                label_subjectivity,
            )

    logger.info(f"Analyzed sentiment for {len(df_out)} rows in '{target_column}'.")
    return df_out


def analyze_sentiment_text(text: str) -> dict:
    """Analyze sentiment of a single text string using TextBlob.

    Args:
        text (str): Input text.

    Returns:
        dict: Sentiment metrics with 'polarity' and 'subjectivity'.

    """
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
    }


def label_polarity(polarity: float) -> str:
    """Convert numeric polarity into categorical labels.

    Args:
        polarity (float): Polarity score from -1.0 to 1.0.

    Returns:
        str: 'positive', 'negative', or 'neutral'.

    """
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"


def label_subjectivity(subjectivity: float) -> str:
    """Convert numeric subjectivity into categorical labels.

    Args:
        subjectivity (float): Subjectivity score from 0.0 to 1.0.

    Returns:
        str: 'subjective' or 'objective'.

    """
    return "subjective" if subjectivity >= 0.5 else "objective"
