"""
Unit tests for Dataset loading, validation, and preprocessing edge cases.
"""

import os
import unittest
import tempfile
import pandas as pd
from train_model import load_dataset


class TestDatasetValidation(unittest.TestCase):

    def test_missing_dataset_file_raises_error(self):
        """Test that attempting to load a nonexistent dataset file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_dataset("nonexistent_path/fake_dataset.csv")

    def test_empty_dataset_file_raises_error(self):
        """Test that an empty CSV file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                load_dataset(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_missing_required_columns_raises_error(self):
        """Test that a CSV missing review or sentiment columns raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df = pd.DataFrame({
                "foo": ["Sample text", "Another text"],
                "bar": [1, 0]
            })
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                load_dataset(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_single_class_dataset_raises_error(self):
        """Test that a dataset containing only one class raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df = pd.DataFrame({
                "review": [f"Positive review number {i}" for i in range(25)],
                "sentiment": ["positive"] * 25
            })
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with self.assertRaises(ValueError):
                load_dataset(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_valid_dataset_loads_correctly(self):
        """Test loading the default valid imdb_dataset.csv."""
        df, rev_col, lbl_col = load_dataset()
        self.assertGreaterEqual(len(df), 20)
        self.assertIn("positive", df[lbl_col].values)
        self.assertIn("negative", df[lbl_col].values)


if __name__ == "__main__":
    unittest.main()
