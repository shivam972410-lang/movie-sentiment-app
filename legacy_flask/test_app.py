"""
Unit and integration tests for CinePulse Movie Sentiment Analysis App.
"""

import unittest
import json
from app import app
from model import analyze_sentiment


class TestMovieSentimentApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page_loads(self):
        """Test that the home page renders correctly with status 200."""
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"CinePulse Sentiment AI", response.data)
        self.assertIn(b"Analyze Sentiment", response.data)

    def test_samples_endpoint(self):
        """Test that /samples returns sample reviews."""
        response = self.app.get("/samples")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["samples"]), 0)

    def test_stats_endpoint(self):
        """Test that /stats returns valid model metadata."""
        response = self.app.get("/stats")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("vocabulary_size", data)
        self.assertIn("TF-IDF", data["model_type"])

    def test_predict_positive_review(self):
        """Test prediction for a clear positive review."""
        payload = {
            "review": "An absolute masterpiece of cinema! The direction and acting were sensational and brilliant."
        }
        response = self.app.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        res = data["data"]
        self.assertEqual(res["sentiment"], "Positive")
        self.assertGreater(res["positive_score"], 50.0)
        self.assertGreater(len(res["top_words"]), 0)

    def test_predict_negative_review(self):
        """Test prediction for a clear negative review."""
        payload = {
            "review": "A complete waste of time. The script is awful, boring, and the acting is terrible."
        }
        response = self.app.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        res = data["data"]
        self.assertEqual(res["sentiment"], "Negative")
        self.assertGreater(res["negative_score"], 50.0)
        self.assertGreater(len(res["top_words"]), 0)

    def test_predict_empty_review(self):
        """Test prediction edge case with empty review string."""
        payload = {"review": "   "}
        response = self.app.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")

    def test_form_post_fallback(self):
        """Test classic form POST request to /."""
        response = self.app.post(
            "/",
            data={"review": "Outstanding performance and beautiful cinematography!"},
            follow_redirects=True
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Positive", response.data)


if __name__ == "__main__":
    unittest.main()
