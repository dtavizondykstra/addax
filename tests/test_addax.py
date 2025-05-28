from addax import addax

"""
Test suite for addax.py - A DataFrame Processing Library
"""
import unittest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys
import io

# Import the module to test
import addax


class TestAddax(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Sample data for testing
        self.sample_data = {
            "Product Name": ["iPhone 14", "Samsung Galaxy", "Google Pixel"],
            "Price ($)": [999, 899, 699],
            "Review Text": [
                "Amazing phone! Love it so much!",
                "Decent phone but battery could be better.",
                "Great camera quality but expensive.",
            ],
            "Rating": [5, 3, 4],
        }
        self.df = pd.DataFrame(self.sample_data)

        # Create a temporary CSV file for testing
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        )
        self.df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)


class TestReadCSV(TestAddax):

    def test_read_csv_success(self):
        """Test successful CSV reading."""
        result = addax.read_csv(self.temp_file.name)
        pd.testing.assert_frame_equal(result, self.df)

    def test_read_csv_file_not_found(self):
        """Test reading non-existent file returns empty DataFrame."""
        result = addax.read_csv("nonexistent_file.csv")
        self.assertTrue(result.empty)

    def test_read_csv_invalid_format(self):
        """Test reading invalid CSV format returns empty DataFrame."""
        # Create invalid CSV file
        invalid_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".csv"
        )
        invalid_file.write('invalid,csv,content\nwith,unclosed,quote"')
        invalid_file.close()

        try:
            result = addax.read_csv(invalid_file.name)
            # Should return empty DataFrame on error
            self.assertTrue(isinstance(result, pd.DataFrame))
        finally:
            os.unlink(invalid_file.name)


class TestStandardizeHeaders(TestAddax):

    def test_standardize_headers_basic(
        self,
    ):  # why not use TestAddax copy of class setup instead
        """Test basic header standardization."""
        df_test = pd.DataFrame(
            {
                "Product Name": [1, 2, 3],
                "Price ($)": [10, 20, 30],
                "User Rating": [4, 5, 3],
            }
        )
        result = addax.standardize_headers(df_test)
        expected_columns = ["product_name", "price_", "user_rating"]
        self.assertEqual(list(result.columns), expected_columns)

    def test_standardize_headers_special_chars(self):
        """Test header standardization with special characters."""
        df_test = pd.DataFrame(
            {"Product@Name!": [1], "Price#($)%": [10], "User   Rating***": [4]}
        )
        result = addax.standardize_headers(df_test)
        expected_columns = ["productname", "price", "user_rating"]
        self.assertEqual(list(result.columns), expected_columns)

    def test_standardize_headers_empty_dataframe(self):
        """Test standardization with empty DataFrame."""
        df_empty = pd.DataFrame()
        result = addax.standardize_headers(df_empty)
        self.assertTrue(result.empty)

    def test_standardize_headers_preserves_data(self):
        """Test that data is preserved during header standardization."""
        original_data = self.df.copy()
        result = addax.standardize_headers(self.df)
        # Check that data values are preserved
        self.assertEqual(len(result), len(original_data))
        self.assertEqual(result.iloc[0, 0], original_data.iloc[0, 0])


class TestFormatTextDataTarget(TestAddax):

    def test_standardize_target_col_data_success(self):
        """Test successful text formatting."""
        df_test = pd.DataFrame(
            {"text_col": ["Hello World!", "Test@Text#123", "Another$Example%"]}
        )
        result = addax.standardize_target_col_data(df_test, "text_col")
        expected = ["hello world", "testtext", "anotherexample"]
        self.assertEqual(list(result["text_col"]), expected)

    def test_standardize_target_col_data_missing_column(self):
        """Test formatting with non-existent column."""
        result = addax.standardize_target_col_data(self.df, "nonexistent_column")
        pd.testing.assert_frame_equal(result, self.df)

    def test_standardize_target_col_data_with_nan(self):
        """Test formatting with NaN values - rows with NaN should be removed."""
        df_test = pd.DataFrame({"text_col": ["Hello World!", None, "Test@Text"]})
        result = addax.standardize_target_col_data(df_test, "text_col")

        # Row with NaN should be removed, leaving only 2 rows
        self.assertEqual(len(result), 2)

        # Check that the remaining values are correctly formatted
        expected_values = ["hello world", "testtext"]
        self.assertEqual(list(result["text_col"]), expected_values)

        # Verify that None/NaN values are not present
        self.assertFalse(result["text_col"].isna().any())

        # Ensure the original DataFrame is unchanged
        self.assertEqual(len(df_test), 3)  # Original should still have 3 rows

    def test_standardize_target_col_data_numeric_values(self):
        """Test formatting with numeric values."""
        df_test = pd.DataFrame({"text_col": [123, 45.67, "89@#$"]})
        result = addax.standardize_target_col_data(df_test, "text_col")
        expected = []
        self.assertEqual(list(result["text_col"]), expected)


class TestRemoveRowsMissingTarget(TestAddax):

    def test_remove_rows_missing_target_success(self):
        """Test successful removal of rows with missing target values."""
        df_test = pd.DataFrame({"col1": [1, 2, 3, 4], "target": ["a", None, "c", "d"]})
        result = addax.remove_rows_missing_target(df_test, "target")
        expected_length = 3  # One row should be removed
        self.assertEqual(len(result), expected_length)
        self.assertNotIn(None, result["target"].values)

    def test_remove_rows_missing_target_no_missing(self):
        """Test with no missing values in target column."""
        df_test = pd.DataFrame({"col1": [1, 2, 3], "target": ["a", "b", "c"]})
        result = addax.remove_rows_missing_target(df_test, "target")
        pd.testing.assert_frame_equal(result, df_test)

    def test_remove_rows_missing_target_all_missing(self):
        """Test with all values missing in target column."""
        df_test = pd.DataFrame({"col1": [1, 2, 3], "target": [None, None, None]})
        result = addax.remove_rows_missing_target(df_test, "target")
        self.assertEqual(len(result), 0)

    def test_remove_rows_missing_target_column_not_found(self):
        """Test with non-existent target column."""
        result = addax.remove_rows_missing_target(self.df, "nonexistent_column")
        pd.testing.assert_frame_equal(result, self.df)


class TestAnalyzeSentimentText(TestAddax):

    def test_analyze_sentiment_text_positive(self):
        """Test sentiment analysis for positive text."""
        text = "I love this product! It's amazing!"
        result = addax.analyze_sentiment_text(text)
        self.assertIn("polarity", result)
        self.assertIn("subjectivity", result)
        self.assertGreater(result["polarity"], 0)  # Should be positive
        self.assertIsInstance(result["polarity"], float)
        self.assertIsInstance(result["subjectivity"], float)

    def test_analyze_sentiment_text_negative(self):
        """Test sentiment analysis for negative text."""
        text = "This product is terrible! I hate it!"
        result = addax.analyze_sentiment_text(text)
        self.assertLess(result["polarity"], 0)  # Should be negative

    def test_analyze_sentiment_text_neutral(self):
        """Test sentiment analysis for neutral text."""
        text = "This is a product."
        result = addax.analyze_sentiment_text(text)
        self.assertAlmostEqual(
            result["polarity"], 0, delta=0.2
        )  # Should be near neutral

    def test_analyze_sentiment_text_empty(self):
        """Test sentiment analysis for empty text."""
        result = addax.analyze_sentiment_text("")
        self.assertEqual(result["polarity"], 0.0)
        self.assertEqual(result["subjectivity"], 0.0)


class TestLabelPolarity(TestAddax):  # all failed

    def test_label_polarity_positive(self):
        """Test polarity labeling for positive values."""
        self.assertEqual(addax.label_polarity(0.5), "positive")
        self.assertEqual(addax.label_polarity(0.2), "positive")

    def test_label_polarity_negative(self):
        """Test polarity labeling for negative values."""
        self.assertEqual(addax.label_polarity(-0.5), "negative")
        self.assertEqual(addax.label_polarity(-0.2), "negative")

    def test_label_polarity_neutral(self):
        """Test polarity labeling for neutral values."""
        self.assertEqual(addax.label_polarity(0.0), "neutral")
        self.assertEqual(addax.label_polarity(0.05), "neutral")
        self.assertEqual(addax.label_polarity(-0.05), "neutral")

    def test_label_polarity_boundary_values(self):
        """Test polarity labeling for boundary values."""
        self.assertEqual(addax.label_polarity(0.1), "neutral")  # exactly 0.1
        self.assertEqual(addax.label_polarity(0.11), "positive")  # just above
        self.assertEqual(addax.label_polarity(-0.1), "neutral")  # exactly -0.1
        self.assertEqual(addax.label_polarity(-0.11), "negative")  # just below


class TestLabelSubjectivity(TestAddax):  # all failed

    def test_label_subjectivity_subjective(self):
        """Test subjectivity labeling for subjective values."""
        self.assertEqual(addax.label_subjectivity(0.7), "subjective")
        self.assertEqual(addax.label_subjectivity(1.0), "subjective")
        self.assertEqual(addax.label_subjectivity(0.5), "subjective")  # boundary

    def test_label_subjectivity_objective(self):
        """Test subjectivity labeling for objective values."""
        self.assertEqual(addax.label_subjectivity(0.3), "objective")
        self.assertEqual(addax.label_subjectivity(0.0), "objective")
        self.assertEqual(addax.label_subjectivity(0.49), "objective")


class TestAnalyzeSentimentDataFrame(TestAddax):

    def test_analyze_sentiment_dataframe_basic(self):
        """Test basic sentiment analysis on DataFrame."""
        df_test = pd.DataFrame(
            {"text": ["I love this!", "This is bad.", "Neutral text."]}
        )
        result = addax.analyze_sentiment_dataframe(df_test, "text")

        self.assertIn("polarity", result.columns)
        self.assertIn("subjectivity", result.columns)
        self.assertEqual(len(result), 3)

    def test_analyze_sentiment_dataframe_with_labels(self):
        """Test sentiment analysis with categorical labels."""
        df_test = pd.DataFrame(
            {"text": ["I love this!", "This is terrible!", "Neutral text."]}
        )
        result = addax.analyze_sentiment_dataframe(df_test, "text", label=True)

        self.assertIn("polarity_label", result.columns)
        self.assertIn("subjectivity_label", result.columns)
        self.assertTrue(
            all(
                label in ["positive", "negative", "neutral"]
                for label in result["polarity_label"]
            )
        )

    def test_analyze_sentiment_dataframe_without_subjectivity(self):
        """Test sentiment analysis without subjectivity."""
        df_test = pd.DataFrame({"text": ["I love this!", "This is bad."]})
        result = addax.analyze_sentiment_dataframe(
            df_test, "text", include_subjectivity=False
        )

        self.assertIn("polarity", result.columns)
        self.assertNotIn("subjectivity", result.columns)

    def test_analyze_sentiment_dataframe_missing_column(self):
        """Test sentiment analysis with non-existent column."""
        with self.assertRaises(ValueError):
            addax.analyze_sentiment_dataframe(self.df, "nonexistent_column")

    def test_analyze_sentiment_dataframe_with_nan(self):  # conflict?
        """Test sentiment analysis with NaN values."""
        df_test = pd.DataFrame({"text": ["I love this!", None, "This is bad."]})
        result = addax.analyze_sentiment_dataframe(df_test, "text")

        # Should handle NaN by filling with empty string
        self.assertEqual(len(result), 3)
        self.assertFalse(result["polarity"].isna().any())


class TestProcessSentiment(TestAddax):

    def test_process_sentiment_full_pipeline(self):
        """Test the complete text processing pipeline."""
        df_test = pd.DataFrame(
            {
                "Product Name": ["iPhone", "Samsung"],
                "Review Text": ["Amazing product!", "Not great quality."],
                "Rating": [5, 2],
            }
        )

        result = addax.process_sentiment(df_test, "Review Text")

        # Check that headers are standardized
        self.assertIn("review_text", result.columns)
        # Check that sentiment analysis was performed
        self.assertIn("polarity", result.columns)
        self.assertIn("polarity_label", result.columns)

    def test_process_sentiment_with_missing_data(self):
        """Test pipeline with missing data in target column."""
        df_test = pd.DataFrame(
            {
                "Product Name": ["iPhone", "Samsung", "Google"],
                "Review Text": ["Amazing product!", None, "Good phone."],
                "Rating": [5, 2, 4],
            }
        )

        result = addax.process_sentiment(df_test, "Review Text")

        # Should have removed the row with missing review text
        self.assertEqual(len(result), 2)

    def test_process_sentiment_case_insensitive(self):
        """Test that column name matching is case insensitive."""
        df_test = pd.DataFrame(
            {"REVIEW TEXT": ["Amazing product!", "Not great quality."]}
        )

        result = addax.process_sentiment(df_test, "Review Text")

        # Should work despite different case
        self.assertIn("review_text", result.columns)
        self.assertIn("polarity", result.columns)


class TestLogging(TestAddax):

    @patch("addax.addax.logger")
    def test_logging_read_csv_success(self, mock_logger):
        """Test that successful CSV reading logs info."""
        addax.read_csv(self.temp_file.name)
        mock_logger.info.assert_called()

    @patch("addax.addax.logger")
    def test_logging_read_csv_error(self, mock_logger):
        """Test that CSV reading error logs error."""
        addax.read_csv("nonexistent_file.csv")
        mock_logger.error.assert_called()

    @patch("addax.addax.logger")
    def test_logging_standardize_headers(self, mock_logger):
        """Test that header standardization logs info."""
        addax.standardize_headers(self.df)
        mock_logger.info.assert_called()


class TestEdgeCases(TestAddax):

    def test_empty_dataframe(self):
        """Test functions with empty DataFrame."""
        df_empty = pd.DataFrame()

        # Most functions should handle empty DataFrames gracefully
        result_std = addax.standardize_headers(df_empty)
        self.assertTrue(result_std.empty)

        result_format = addax.standardize_target_col_data(df_empty, "text")
        self.assertTrue(result_format.empty)

        result_remove = addax.remove_rows_missing_target(df_empty, "text")
        self.assertTrue(result_remove.empty)

    def test_dataframe_with_special_characters(self):
        """Test handling of DataFrames with special characters."""
        df_special = pd.DataFrame({"text_col": ["café", "naïve", "résumé", "你好"]})

        result = addax.standardize_target_col_data(df_special, "text_col")
        # Should handle unicode characters
        self.assertEqual(len(result), 4)

    def test_very_long_text(self):
        """Test sentiment analysis with very long text."""
        long_text = "This is great! " * 1000  # Very long positive text
        result = addax.analyze_sentiment_text(long_text)

        self.assertIsInstance(result["polarity"], float)
        self.assertIsInstance(result["subjectivity"], float)
        self.assertGreater(result["polarity"], 0)  # Should still be positive


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
