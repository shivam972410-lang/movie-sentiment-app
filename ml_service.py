"""
ML Service module for Movie Sentiment Analysis.
Provides consistent text cleaning, model loading, explainable prediction,
and threshold-based uncertainty handling.
"""

import os
import re
import json
import joblib
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.joblib")
DEFAULT_METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# In-memory model cache
_CACHED_MODEL = None

# Built-in standard English stopwords list (fallback if NLTK is not downloaded)
FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few",
    "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "me", "more", "most", "my", "myself", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your",
    "yours", "yourself", "yourselves"
}

# Preserve negation words that are critical for sentiment
SENTIMENT_PRESERVE_WORDS = {
    "not", "no", "nor", "neither", "never", "none", "hardly", "scarcely", "barely"
}


def get_stopwords() -> set:
    """Attempts to use NLTK stopwords if available, otherwise uses built-in set."""
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            sw = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            sw = set(stopwords.words("english"))
        # Remove crucial negation words from stopwords
        return sw - SENTIMENT_PRESERVE_WORDS
    except Exception:
        return FALLBACK_STOPWORDS - SENTIMENT_PRESERVE_WORDS


def clean_text(text: str, remove_stop_words: bool = True) -> str:
    """
    Standard text preprocessor used identically in both training and inference.
    - Strips HTML tags (e.g., '<br />')
    - Strips URLs
    - Normalizes characters and punctuation
    - Converts to lowercase
    - Optionally removes non-sentiment stopwords
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove HTML markup
    cleaned = re.sub(r"<[^>]+>", " ", text)
    # Remove URL links
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    # Convert to lowercase
    cleaned = cleaned.lower()
    # Replace non-alphanumeric chars (except basic punctuation / apostrophes) with spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s']", " ", cleaned)
    # Collapse multiple whitespaces
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", cleaned)

    if remove_stop_words:
        stop_words = get_stopwords()
        tokens = [w for w in tokens if w not in stop_words or w in SENTIMENT_PRESERVE_WORDS]

    return " ".join(tokens)


def load_model(model_path: str = DEFAULT_MODEL_PATH, force_reload: bool = False) -> dict:
    """
    Loads and caches the model payload from disk.
    Raises clear FileNotFoundError if the model has not been trained yet.
    """
    global _CACHED_MODEL
    # If custom model path is requested, do not use default cache
    if model_path != DEFAULT_MODEL_PATH:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model artifact not found at '{model_path}'. "
                "Please train the model first by running: python train_model.py"
            )
        return joblib.load(model_path)

    if _CACHED_MODEL is not None and not force_reload:
        return _CACHED_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model artifact not found at '{model_path}'. "
            "Please train the model first by running: python train_model.py"
        )

    payload = joblib.load(model_path)
    _CACHED_MODEL = payload
    return payload


def load_metrics(metrics_path: str = DEFAULT_METRICS_PATH) -> dict:
    """
    Loads saved evaluation metrics from disk.
    Raises FileNotFoundError if metrics file does not exist.
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_path}'. "
            "Please run 'python train_model.py' to generate actual evaluation metrics."
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_sentiment(
    text: str,
    confidence_threshold: float = 0.60,
    model_path: str = DEFAULT_MODEL_PATH
) -> dict:
    """
    Predicts sentiment for a movie review using TF-IDF + Logistic Regression.

    Returns:
    - review: Original text
    - cleaned_review: Sanitized text
    - sentiment: Final label ('Positive', 'Negative', or 'Neutral / Uncertain')
    - raw_sentiment: Binary model prediction before thresholding
    - confidence: Confidence probability (0.0 to 1.0)
    - positive_probability: Probability of positive class
    - negative_probability: Probability of negative class
    - is_uncertain: Boolean flag if confidence < threshold
    - explanation: Clear rationale for the classification or threshold decision
    - top_keywords: Key terms driving positive/negative influence
    - word_count: Number of words in cleaned text
    - char_count: Length of cleaned text
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review text cannot be empty or whitespace only.")

    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Review does not contain any valid words after preprocessing.")

    model_data = load_model(model_path)
    pipeline = model_data.get("pipeline")
    vectorizer = model_data.get("vectorizer")
    classifier = model_data.get("classifier")
    classes = model_data.get("classes", ["negative", "positive"])

    # If pipeline is stored directly
    if pipeline is not None:
        vectorizer = pipeline.named_steps.get("tfidf", vectorizer)
        classifier = pipeline.named_steps.get("clf", classifier)

    if vectorizer is None or classifier is None:
        raise ValueError("Corrupt model payload: missing vectorizer or classifier.")

    # Vectorize cleaned text
    tfidf_vec = vectorizer.transform([cleaned])

    # Check if any vocabulary matched
    if tfidf_vec.nnz == 0:
        return {
            "review": text,
            "cleaned_review": cleaned,
            "sentiment": "Neutral / Uncertain",
            "raw_sentiment": "Uncertain",
            "confidence": 0.50,
            "positive_probability": 0.50,
            "negative_probability": 0.50,
            "is_uncertain": True,
            "explanation": (
                "None of the words in the review matched the trained IMDb vocabulary. "
                "Labeled as Neutral / Uncertain due to lack of distinct signal."
            ),
            "top_keywords": [],
            "word_count": len(cleaned.split()),
            "char_count": len(cleaned),
        }

    # Predict probabilities
    probs = classifier.predict_proba(tfidf_vec)[0]

    # Map class probabilities
    pos_idx = classes.index("positive") if "positive" in classes else 1
    neg_idx = classes.index("negative") if "negative" in classes else 0

    pos_prob = float(probs[pos_idx])
    neg_prob = float(probs[neg_idx])

    if pos_prob >= neg_prob:
        raw_sentiment = "Positive"
        raw_confidence = pos_prob
    else:
        raw_sentiment = "Negative"
        raw_confidence = neg_prob

    # Apply thresholding
    # The training data only has positive and negative labels.
    # If confidence is below the threshold, label as Neutral / Uncertain.
    is_uncertain = raw_confidence < confidence_threshold
    if is_uncertain:
        sentiment = "Neutral / Uncertain"
        explanation = (
            f"Confidence score of {raw_confidence * 100:.1f}% is below the configured "
            f"decision threshold of {confidence_threshold * 100:.1f}%. "
            "Because the model was trained strictly on binary IMDb positive/negative reviews, "
            "samples close to the 50/50 decision boundary are categorized as Neutral / Uncertain "
            "as a threshold-based decision to prevent false certainty."
        )
    else:
        sentiment = raw_sentiment
        explanation = (
            f"Classified as {raw_sentiment} with a high confidence score of "
            f"{raw_confidence * 100:.1f}%."
        )

    # Word attribution / Explainable AI
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]  # shape: (n_features,)
    nz_indices = tfidf_vec.nonzero()[1]

    top_keywords = []
    for idx in nz_indices:
        word = feature_names[idx]
        tfidf_val = tfidf_vec[0, idx]
        # coefficient positive -> pulls towards positive class; negative -> pulls towards negative
        weight = float(coefficients[idx] * tfidf_val)
        pull = "positive" if weight > 0 else "negative"
        top_keywords.append({
            "term": word,
            "weight": round(abs(weight), 4),
            "sentiment": pull,
            "score": round(weight, 4)
        })

    # Sort keywords by impact magnitude
    top_keywords.sort(key=lambda x: abs(x["score"]), reverse=True)

    words = cleaned.split()
    return {
        "review": text,
        "cleaned_review": cleaned,
        "sentiment": sentiment,
        "raw_sentiment": raw_sentiment,
        "confidence": round(raw_confidence, 4),
        "positive_probability": round(pos_prob, 4),
        "negative_probability": round(neg_prob, 4),
        "is_uncertain": is_uncertain,
        "explanation": explanation,
        "top_keywords": top_keywords[:10],
        "word_count": len(words),
        "char_count": len(cleaned),
    }
