# addax.py
"""Addax: A DataFrame Processing Library"""
import logging
import re
from typing import Optional, List

import pandas as pd
from textblob import TextBlob

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.

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
        logger.error(f"Error reading CSV file '{file_path}': {e}")
        return pd.DataFrame()


def standardize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes DataFrame column headers: lowercases and replaces spaces with underscores.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with standardized column names.
    """
    df_std = df.copy()
    standardized = []
    for col in df_std.columns:
        # lowercase, replace spaces with underscores, remove special chars
        new_col = col.lower()
        new_col = re.sub(r"\s+", "_", new_col)
        new_col = re.sub(r"[^0-9a-zA-Z_]+", "", new_col)
        standardized.append(new_col)
    df_std.columns = standardized
    logger.info("Standardized headers to lowercase and underscores.")
    return df_std


def format_text_data_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Cleans textual data in the specified column by lowercasing and removing special characters.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        target_column (str): Name of the column to format.

    Returns:
        pd.DataFrame: DataFrame with cleaned text in the target column.
    """
    if target_column not in df.columns:
        logger.warning(
            f"Target column '{target_column}' not found. No formatting applied."
        )
        return df.copy()

    df_clean = df.copy()
    df_clean[target_column] = (
        df_clean[target_column]
        .astype(str)
        .str.lower()
        .apply(lambda x: re.sub(r"[^\w\s]", "", x))
    )
    logger.info(
        f"Formatted text in column '{target_column}': lowercased and removed special characters."
    )
    return df_clean


def remove_rows_missing_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Drops rows where the target column has missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Column name to check for missing values.

    Returns:
        pd.DataFrame: Filtered DataFrame without rows missing in target_column.
    """
    if target_column not in df.columns:
        logger.warning(
            f"Target column '{target_column}' not in DataFrame. No rows removed."
        )
        return df.copy()

    df_filtered = df[df[target_column].notna()].copy()
    removed = len(df) - len(df_filtered)
    logger.info(f"Removed {removed} rows with missing values in '{target_column}'.")
    return df_filtered


def analyze_sentiment_text(text: str) -> dict:
    """
    Analyzes sentiment of a single text string using TextBlob.

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
    """
    Converts numeric polarity into categorical labels.

    Args:
        polarity (float): Polarity score from -1.0 to 1.0.

    Returns:
        str: 'positive', 'negative', or 'neutral'.
    """
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"


def label_subjectivity(subjectivity: float) -> str:
    """
    Converts numeric subjectivity into categorical labels.

    Args:
        subjectivity (float): Subjectivity score from 0.0 to 1.0.

    Returns:
        str: 'subjective' or 'objective'.
    """
    return "subjective" if subjectivity >= 0.5 else "objective"


def analyze_sentiment_dataframe(
    df: pd.DataFrame,
    text_column: str,
    include_subjectivity: bool = True,
    label: bool = False,
) -> pd.DataFrame:
    """
    Applies sentiment analysis over a DataFrame column of text, with optional labels.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the column containing text.
        include_subjectivity (bool): Whether to include subjectivity score.
        label (bool): Whether to add categorical labels for polarity and subjectivity.

    Returns:
        pd.DataFrame: DataFrame with 'polarity', optional 'subjectivity', and optional 'polarity_label' and 'subjectivity_label'.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    df_out = df.copy()
    df_out["polarity"] = (
        df_out[text_column]
        .fillna("")
        .apply(lambda t: analyze_sentiment_text(t)["polarity"])
    )

    if include_subjectivity:
        df_out["subjectivity"] = (
            df_out[text_column]
            .fillna("")
            .apply(lambda t: analyze_sentiment_text(t)["subjectivity"])
        )

    if label:
        df_out["polarity_label"] = df_out["polarity"].apply(label_polarity)
        if include_subjectivity:
            df_out["subjectivity_label"] = df_out["subjectivity"].apply(
                label_subjectivity
            )

    logger.info(f"Analyzed sentiment for {len(df_out)} rows in '{text_column}'.")
    return df_out


def process_text_column(
    df: pd.DataFrame,
    text_column: str,
    include_subjectivity: bool = True,
    label: bool = False,
) -> pd.DataFrame:
    """
    End-to-end pipeline: standardizes headers, formats the target text column,
    removes missing entries, and runs sentiment analysis with optional labeling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_column (str): Name of the text column to process.
        include_subjectivity (bool): Whether to include subjectivity score.
        label (bool): Whether to add categorical labels.

    Returns:
        pd.DataFrame: Processed DataFrame with 'polarity', optional 'subjectivity', and optional labels.
    """
    df_proc = standardize_headers(df)
    std_col = text_column.lower().replace(" ", "_")
    df_proc = format_text_data_target(df_proc, std_col)
    df_proc = remove_rows_missing_target(df_proc, std_col)
    df_proc = analyze_sentiment_dataframe(
        df_proc, std_col, include_subjectivity=include_subjectivity, label=True
    )
    return df_proc
