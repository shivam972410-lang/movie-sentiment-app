"""
API Integration tests for FastAPI backend endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from main import app


class TestFastAPIBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Test GET / returns online status and endpoint map."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("endpoints", data)

    def test_health_check_endpoint(self):
        """Test GET /health reports model readiness."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("model_loaded", data)
        self.assertTrue(data["model_loaded"])
        self.assertIn("integrations", data)

    def test_predict_positive(self):
        """Test POST /predict with a strong positive review."""
        payload = {"review": "An extraordinary achievement in filmmaking! Deeply resonant and brilliant."}
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["sentiment"], "Positive")
        self.assertGreater(data["positive_probability"], 0.50)
        self.assertIsInstance(data["top_keywords"], list)

    def test_predict_negative(self):
        """Test POST /predict with an overtly negative review."""
        payload = {"review": "A complete and utter waste of time. The script makes zero sense and acting was awful."}
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["sentiment"], "Negative")
        self.assertGreater(data["negative_probability"], 0.50)

    def test_predict_empty_input_returns_400(self):
        """Test POST /predict with blank string returns 400."""
        payload = {"review": "   "}
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 400)

    def test_predict_threshold_neutral(self):
        """Test POST /predict with high threshold returns Neutral / Uncertain."""
        payload = {
            "review": "The cinematography was fine but dialogue felt somewhat unusual.",
            "confidence_threshold": 0.95
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["sentiment"], "Neutral / Uncertain")
        self.assertTrue(data["is_uncertain"])

    def test_predict_batch(self):
        """Test POST /predict/batch with multiple reviews."""
        payload = {
            "reviews": [
                "An absolute cinematic masterpiece with breathtaking directing.",
                "Dull, lifeless, and painfully boring.",
                "Visually stunning sci-fi spectacle with great philosophical themes."
            ]
        }
        resp = self.client.post("/predict/batch", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["total_reviews"], 3)
        self.assertIn("summary", data)
        self.assertEqual(len(data["results"]), 3)

    def test_predict_batch_empty_list_returns_400(self):
        """Test POST /predict/batch with empty list returns 400 or 422."""
        payload = {"reviews": []}
        resp = self.client.post("/predict/batch", json=payload)
        self.assertIn(resp.status_code, [400, 422])

    def test_movie_endpoint_unconfigured_or_graceful(self):
        """Test GET /movie/{title} responds gracefully without crashing or leaking keys."""
        resp = self.client.get("/movie/Inception")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        # Should be unconfigured (if no key in env) or success/not_found
        self.assertIn(data["status"], ["unconfigured", "success", "not_found", "error"])

    def test_movie_empty_title_returns_400_or_404(self):
        """Test GET /movie/ with whitespace title."""
        resp = self.client.get("/movie/%20%20")
        self.assertIn(resp.status_code, [400, 404])

    def test_chat_endpoint_unconfigured_or_graceful(self):
        """Test POST /chat responds gracefully when Gemini key is unconfigured."""
        payload = {"message": "Recommend top classic sci-fi movies."}
        resp = self.client.post("/chat", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn(data["status"], ["unconfigured", "success", "error"])

    def test_chat_empty_message_returns_400_or_422(self):
        """Test POST /chat with empty message returns error."""
        resp = self.client.post("/chat", json={"message": "  "})
        self.assertIn(resp.status_code, [400, 422])

    def test_metrics_endpoint(self):
        """Test GET /metrics returns valid model performance data."""
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        metrics = data["metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)


if __name__ == "__main__":
    unittest.main()
