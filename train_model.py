"""
Train Model Script for Movie Sentiment Analysis.
Loads an IMDb movie review CSV dataset, preprocesses text consistently,
trains a TF-IDF + Logistic Regression pipeline using an 80/20 stratified split,
computes actual evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix),
and persists the fitted pipeline and metrics.
"""

import os
import sys
import json
import argparse
import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Import shared text cleaning from ml_service
from ml_service import clean_text, BASE_DIR, MODELS_DIR, DEFAULT_MODEL_PATH, DEFAULT_METRICS_PATH

DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATASET_CANDIDATES = [
    os.path.join(DEFAULT_DATA_DIR, "imdb_dataset.csv"),
    os.path.join(DEFAULT_DATA_DIR, "IMDB Dataset.csv"),
    os.path.join(DEFAULT_DATA_DIR, "imdb_reviews.csv"),
]

# Accepted column aliases
REVIEW_COL_ALIASES = ["review", "text", "review_text", "reviews", "content", "document"]
LABEL_COL_ALIASES = ["sentiment", "label", "target", "class", "rating_class"]


def find_column(df: pd.DataFrame, aliases: list) -> str:
    """Finds matching column name case-insensitively."""
    lower_cols = {str(col).strip().lower(): col for col in df.columns}
    for alias in aliases:
        if alias.lower() in lower_cols:
            return lower_cols[alias.lower()]
    return None


def load_dataset(csv_path: str = None) -> tuple[pd.DataFrame, str, str]:
    """
    Loads dataset from provided path or default candidates.
    Validates file existence, non-emptiness, and required columns.
    Returns: (df, review_column, label_column)
    """
    selected_path = None
    if csv_path:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Specified dataset file does not exist: '{csv_path}'")
        selected_path = csv_path
    else:
        for candidate in DEFAULT_DATASET_CANDIDATES:
            if os.path.exists(candidate):
                selected_path = candidate
                break

    if not selected_path:
        raise FileNotFoundError(
            f"No dataset file found in '{DEFAULT_DATA_DIR}'. "
            f"Please place an IMDb CSV file (e.g., 'imdb_dataset.csv' or 'IMDB Dataset.csv') in '{DEFAULT_DATA_DIR}' "
            "or specify path with '--dataset <path>'."
        )

    print(f"Loading dataset from: {selected_path}")
    try:
        df = pd.read_csv(selected_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV dataset at '{selected_path}': {e}") from e

    if df.empty:
        raise ValueError(f"The dataset at '{selected_path}' is completely empty.")

    # Detect review column
    review_col = find_column(df, REVIEW_COL_ALIASES)
    if not review_col:
        raise ValueError(
            f"Dataset missing required review text column. "
            f"Found columns: {list(df.columns)}. Expected one of: {REVIEW_COL_ALIASES}"
        )

    # Detect label column
    label_col = find_column(df, LABEL_COL_ALIASES)
    if not label_col:
        raise ValueError(
            f"Dataset missing required sentiment label column. "
            f"Found columns: {list(df.columns)}. Expected one of: {LABEL_COL_ALIASES}"
        )

    # Clean and filter labels
    df = df.dropna(subset=[review_col, label_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    # Map numerical or binary labels if necessary (e.g. 1 -> positive, 0 -> negative)
    label_mapping = {
        "pos": "positive", "1": "positive", "1.0": "positive",
        "neg": "negative", "0": "negative", "0.0": "negative"
    }
    df[label_col] = df[label_col].replace(label_mapping)

    # Filter to only positive and negative
    valid_df = df[df[label_col].isin(["positive", "negative"])].copy()

    unique_labels = valid_df[label_col].unique()
    if len(unique_labels) < 2:
        raise ValueError(
            f"Dataset must contain both 'positive' and 'negative' sentiment classes. "
            f"Found labels: {list(unique_labels)}"
        )

    if len(valid_df) < 20:
        raise ValueError(
            f"Dataset contains only {len(valid_df)} valid samples. "
            "A minimum of 20 samples is required for stratified train/test splitting."
        )

    print(f"Dataset successfully loaded: {len(valid_df)} valid reviews.")
    print(f"Class distribution: {dict(valid_df[label_col].value_counts())}")

    return valid_df, review_col, label_col


def train_pipeline(
    csv_path: str = None,
    output_model_path: str = DEFAULT_MODEL_PATH,
    output_metrics_path: str = DEFAULT_METRICS_PATH,
    test_size: float = 0.20,
    random_state: int = 42
) -> dict:
    """
    Executes the end-to-end training and evaluation pipeline:
    1. Loads dataset
    2. Cleans text consistently with ml_service.clean_text
    3. Stratified 80/20 train/test split
    4. Fits TF-IDF + Logistic Regression pipeline
    5. Calculates actual metrics on test set
    6. Saves pipeline and metrics JSON
    """
    df, review_col, label_col = load_dataset(csv_path)

    print("\n--- Preprocessing Reviews ---")
    df["clean_review"] = df[review_col].astype(str).apply(clean_text)
    # Remove any rows where cleaning produced empty strings
    df = df[df["clean_review"].str.len() > 3].copy()

    X = df["clean_review"]
    y = df[label_col]

    print(f"Reviews remaining after preprocessing: {len(df)}")

    # Stratified split: exactly 80% train, 20% test
    print(f"\n--- Splitting Dataset ({int((1-test_size)*100)}/{int(test_size*100)} Stratified Split) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Training set: {len(X_train)} samples ({dict(y_train.value_counts())})")
    print(f"Testing set:  {len(X_test)} samples ({dict(y_test.value_counts())})")

    # Build Pipeline: TF-IDF Vectorizer + Logistic Regression
    print("\n--- Training TF-IDF & Logistic Regression Pipeline ---")
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=15000,
                sublinear_tf=True,
                min_df=1
            )
        ),
        (
            "clf",
            LogisticRegression(
                C=2.0,
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced"
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on holdout test set (NO FABRICATED METRICS)
    print("\n--- Evaluating on Test Set ---")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_binary = float(precision_score(y_test, y_pred, pos_label="positive", zero_division=0))
    rec_binary = float(recall_score(y_test, y_pred, pos_label="positive", zero_division=0))
    f1_binary = float(f1_score(y_test, y_pred, pos_label="positive", zero_division=0))

    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"]).tolist()
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec_binary * 100:.2f}% (positive class)")
    print(f"Recall:    {rec_binary * 100:.2f}% (positive class)")
    print(f"F1-Score:  {f1_binary * 100:.2f}% (positive class)")
    print(f"Macro F1:  {f1_macro * 100:.2f}%")
    print(f"Confusion Matrix (labels=['negative', 'positive']): {cm}")

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]

    # Package model payload
    model_payload = {
        "pipeline": pipeline,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "classes": classifier.classes_.tolist(),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": acc,
        "f1_score": f1_binary,
    }

    # Prepare metrics payload for serialization
    metrics_payload = {
        "model_type": "TF-IDF (Unigrams + Bigrams) + Logistic Regression",
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_name": os.path.basename(csv_path or DEFAULT_DATASET_CANDIDATES[0]),
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "split_ratio": f"{int((1-test_size)*100)}/{int(test_size*100)}",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "ngram_range": [1, 2],
        "classes": classifier.classes_.tolist(),
        "accuracy": round(acc, 4),
        "precision": round(prec_binary, 4),
        "recall": round(rec_binary, 4),
        "f1_score": round(f1_binary, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "confusion_matrix": cm,
        "classification_report": report_dict,
    }

    # Save artifacts
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)

    joblib.dump(model_payload, output_model_path)
    print(f"\nModel pipeline saved to: {output_model_path}")

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Metrics saved to: {output_metrics_path}")

    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Train Movie Sentiment Analysis Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to CSV dataset containing movie reviews and sentiment labels"
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to save trained joblib model"
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=DEFAULT_METRICS_PATH,
        help="Path to save evaluation metrics JSON"
    )
    args = parser.parse_args()

    try:
        train_pipeline(
            csv_path=args.dataset,
            output_model_path=args.model_out,
            output_metrics_path=args.metrics_out
        )
        print("\nTraining completed successfully! Ready for FastAPI and Streamlit.")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
