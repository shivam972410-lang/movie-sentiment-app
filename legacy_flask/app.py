"""
Flask Application for Movie Sentiment Analysis.
Provides web dashboard and REST API for real-time sentiment classification.
"""

import os
from flask import Flask, render_template, request, jsonify
from model import analyze_sentiment, get_model

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)

# Pre-defined curated sample reviews for quick 1-click user testing
PRESET_SAMPLES = [
    {
        "id": "masterpiece",
        "title": "Cinematic Masterpiece",
        "badge": "Positive",
        "text": "An absolute cinematic masterpiece! Christopher Nolan delivers breathtaking visuals, exceptional sound design, and an unforgettable climax. The acting by the entire ensemble cast was simply phenomenal."
    },
    {
        "id": "flop",
        "title": "Box Office Flop",
        "badge": "Negative",
        "text": "A complete and utter waste of time. The script makes zero sense, the pacing is agonizingly slow, and the dialogue was painfully cringeworthy. One of the worst movies I have ever had the misfortune of watching."
    },
    {
        "id": "sci-fi-thriller",
        "title": "Thrilling Sci-Fi",
        "badge": "Positive",
        "text": "Visually stunning sci-fi spectacle with thought-provoking philosophical themes. The stunt choreography was exhilarating and the musical score brought genuine goosebumps."
    },
    {
        "id": "lazy-sequel",
        "title": "Disappointing Sequel",
        "badge": "Negative",
        "text": "A cheap cash-grab sequel with zero soul, terrible sound mixing, and hideous CGI. Predictable, uninspired, and completely devoid of any originality or charm."
    },
    {
        "id": "charming-comedy",
        "title": "Heartwarming Gem",
        "badge": "Positive",
        "text": "Smart, funny, charming, and thoroughly entertaining. The chemistry between the leads was electric, and the witty dialogue kept a smile on my face from beginning to end."
    }
]


@app.route("/", methods=["GET", "POST"])
def index():
    initial_result = None
    initial_review = ""
    if request.method == "POST":
        initial_review = request.form.get("review", "")
        if initial_review.strip():
            initial_result = analyze_sentiment(initial_review)

    return render_template(
        "index.html",
        samples=PRESET_SAMPLES,
        initial_result=initial_result,
        initial_review=initial_review
    )


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for asynchronous sentiment prediction."""
    data = request.get_json(silent=True) or request.form
    review_text = data.get("review", "").strip()

    if not review_text:
        return jsonify({
            "status": "error",
            "message": "Please enter a movie review to analyze."
        }), 400

    result = analyze_sentiment(review_text)
    return jsonify({
        "status": "success",
        "data": result
    })


@app.route("/samples", methods=["GET"])
def get_samples():
    """Returns curated preset reviews for quick exploration."""
    return jsonify({
        "status": "success",
        "samples": PRESET_SAMPLES
    })


@app.route("/stats", methods=["GET"])
def get_stats():
    """Returns model metadata and performance metrics."""
    try:
        model_data = get_model()
        vectorizer = model_data["vectorizer"]
        vocab_size = len(vectorizer.vocabulary_)
        accuracy = round(model_data.get("accuracy", 0.9) * 100, 1)
        total_samples = model_data.get("total_samples", 0)

        return jsonify({
            "status": "success",
            "model_type": "TF-IDF + Logistic Regression",
            "ngram_range": "Unigrams + Bigrams (1, 2)",
            "vocabulary_size": vocab_size,
            "test_accuracy": f"{accuracy}%",
            "training_samples": total_samples,
            "dataset": "Kaggle IMDB Movie Reviews Format"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    # Ensure model exists
    get_model()
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting Movie Sentiment Analysis server at http://127.0.0.1:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
