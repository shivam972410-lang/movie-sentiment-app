"""
Unit tests for Machine Learning model, preprocessing, and inference logic.
"""

import os
import unittest
from ml_service import clean_text, predict_sentiment, load_model, DEFAULT_MODEL_PATH


class TestMLService(unittest.TestCase):

    def test_clean_text_basic(self):
        """Test HTML and URL stripping, lowercase, punctuation normalization."""
        raw = "Check this out <br /><br /> at https://example.com! Truly <b>AMAZING</b> film!!!"
        cleaned = clean_text(raw)
        self.assertNotIn("<br", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertIn("amazing", cleaned)
        self.assertIn("truly", cleaned)

    def test_clean_text_empty_and_whitespace(self):
        """Test clean_text handling of empty or blank strings."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("    "), "")
        self.assertEqual(clean_text(None), "")

    def test_predict_positive_review(self):
        """Test prediction for an overtly positive movie review."""
        review = "An extraordinary achievement in filmmaking! Deeply resonant, emotional, and brilliant."
        result = predict_sentiment(review)
        self.assertEqual(result["sentiment"], "Positive")
        self.assertGreater(result["positive_probability"], 0.50)
        self.assertFalse(result["is_uncertain"])
        self.assertIn("top_keywords", result)
        self.assertGreater(len(result["top_keywords"]), 0)

    def test_predict_negative_review(self):
        """Test prediction for an overtly negative movie review."""
        review = "A complete and utter waste of time. The script makes zero sense and acting was awful."
        result = predict_sentiment(review)
        self.assertEqual(result["sentiment"], "Negative")
        self.assertGreater(result["negative_probability"], 0.50)
        self.assertFalse(result["is_uncertain"])
        self.assertIn("top_keywords", result)
        self.assertGreater(len(result["top_keywords"]), 0)

    def test_predict_empty_review_error(self):
        """Test that empty or whitespace reviews raise ValueError."""
        with self.assertRaises(ValueError):
            predict_sentiment("")
        with self.assertRaises(ValueError):
            predict_sentiment("   ")

    def test_predict_threshold_uncertainty(self):
        """Test that setting a high threshold flags ambiguous reviews as Neutral / Uncertain."""
        review = "The cinematography was reasonably good though the pacing felt slightly off."
        # Using a very high threshold (e.g. 0.95) should trigger the threshold decision
        result = predict_sentiment(review, confidence_threshold=0.95)
        self.assertEqual(result["sentiment"], "Neutral / Uncertain")
        self.assertTrue(result["is_uncertain"])
        self.assertIn("threshold", result["explanation"].lower())

    def test_missing_model_file_error(self):
        """Test that passing a nonexistent model path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            predict_sentiment("Good movie", model_path="nonexistent_model.joblib")


if __name__ == "__main__":
    unittest.main()
