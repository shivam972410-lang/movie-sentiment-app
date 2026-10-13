"""
FastAPI Backend for Movie Sentiment Analysis.
Provides RESTful endpoints for sentiment prediction, batch classification,
OMDb movie metadata retrieval with plot sentiment, and Gemini-powered CineBot chat.
"""

import os
import httpx
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

from ml_service import (
    predict_sentiment,
    load_metrics,
    load_model,
    clean_text,
    DEFAULT_MODEL_PATH,
    DEFAULT_METRICS_PATH
)

# App configuration
API_TITLE = "Movie Sentiment Analysis API"
API_VERSION = "2.0.0"
API_DESCRIPTION = (
    "Production-grade Machine Learning API for movie review sentiment classification, "
    "explainable AI word attribution, OMDb film retrieval with plot sentiment, "
    "and Google Gemini AI cinema consultation."
)

CONFIDENCE_THRESHOLD_ENV = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Request & Response Schemas
# ==========================================

class PredictRequest(BaseModel):
    review: str = Field(..., description="Movie review text to analyze", min_length=1)
    confidence_threshold: Optional[float] = Field(
        default=None,
        description="Threshold below which prediction is marked Neutral/Uncertain (0.50 to 0.95)",
        ge=0.50,
        le=0.95
    )


class KeywordItem(BaseModel):
    term: str
    weight: float
    sentiment: str
    score: float


class PredictResponse(BaseModel):
    status: str = "success"
    review: str
    cleaned_review: str
    sentiment: str
    raw_sentiment: str
    confidence: float
    positive_probability: float
    negative_probability: float
    is_uncertain: bool
    explanation: str
    top_keywords: List[KeywordItem]
    word_count: int
    char_count: int


class BatchPredictRequest(BaseModel):
    reviews: List[str] = Field(..., description="List of review texts", min_length=1)
    confidence_threshold: Optional[float] = Field(default=None, ge=0.50, le=0.95)


class BatchPredictResponse(BaseModel):
    status: str = "success"
    total_reviews: int
    summary: Dict[str, int]
    results: List[PredictResponse]


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant' / 'model'")
    content: str = Field(...)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query about movies or sentiment", min_length=1)
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation turns")


class ChatResponse(BaseModel):
    status: str = "success"
    reply: str
    model_used: str


# ==========================================
# API Endpoints
# ==========================================

@app.get("/", tags=["Status"])
def root():
    """Returns general API information and service status."""
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "movie_lookup": "/movie/{title}",
            "chat": "/chat",
            "metrics": "/metrics"
        }
    }


@app.get("/health", tags=["Status"])
def health_check():
    """Health check reporting model readiness and external service configuration."""
    model_loaded = False
    model_error = None
    try:
        load_model()
        model_loaded = True
    except Exception as e:
        model_error = str(e)

    metrics_available = os.path.exists(DEFAULT_METRICS_PATH)

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_error": model_error,
        "metrics_available": metrics_available,
        "integrations": {
            "omdb_api_configured": bool(OMDB_API_KEY),
            "gemini_api_configured": bool(GEMINI_API_KEY)
        },
        "default_confidence_threshold": CONFIDENCE_THRESHOLD_ENV
    }


@app.post("/predict", response_model=PredictResponse, tags=["Sentiment Analysis"])
def predict(request: PredictRequest):
    """
    Analyzes sentiment of a movie review.
    Outputs Positive, Negative, or threshold-based Neutral/Uncertain with explainability.
    """
    review_text = request.review.strip()
    if not review_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty or only whitespace."
        )

    threshold = request.confidence_threshold if request.confidence_threshold is not None else CONFIDENCE_THRESHOLD_ENV

    try:
        result = predict_sentiment(review_text, confidence_threshold=threshold)
        return {
            "status": "success",
            **result
        }
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf)
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Sentiment Analysis"])
def predict_batch(request: BatchPredictRequest):
    """Batch prediction endpoint for CSV rows or multiple movie reviews."""
    threshold = request.confidence_threshold if request.confidence_threshold is not None else CONFIDENCE_THRESHOLD_ENV

    if not request.reviews:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The reviews list cannot be empty."
        )

    results = []
    summary = {"Positive": 0, "Negative": 0, "Neutral / Uncertain": 0}

    for text in request.reviews:
        cleaned = text.strip() if isinstance(text, str) else ""
        if not cleaned:
            continue
        try:
            pred = predict_sentiment(cleaned, confidence_threshold=threshold)
            results.append(PredictResponse(status="success", **pred))
            sentiment = pred["sentiment"]
            if sentiment in summary:
                summary[sentiment] += 1
            else:
                summary["Neutral / Uncertain"] += 1
        except Exception:
            continue

    return {
        "status": "success",
        "total_reviews": len(results),
        "summary": summary,
        "results": results
    }


@app.get("/movie/{title}", tags=["OMDb Integration"])
async def get_movie_details(title: str):
    """
    Fetches film metadata from OMDb API and analyzes plot sentiment when available.
    Never exposes raw API keys.
    """
    title_clean = title.strip()
    if not title_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Movie title must be specified."
        )

    api_key = os.getenv("OMDB_API_KEY", "").strip() or OMDB_API_KEY
    if not api_key:
        return {
            "status": "unconfigured",
            "message": (
                "OMDb API key is not configured. Please add OMDB_API_KEY to your .env file "
                "to enable live movie data retrieval. You can obtain a free key at https://www.omdbapi.com/apikey.aspx"
            ),
            "movie": None,
            "plot_sentiment": None
        }

    omdb_url = "https://www.omdbapi.com/"
    params = {
        "t": title_clean,
        "plot": "full",
        "apikey": api_key
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(omdb_url, params=params)

        if resp.status_code == 401:
            return {
                "status": "unconfigured",
                "message": (
                    "OMDb returned '401 Unauthorized' (API key inactive or invalid). "
                    "Please check your email and click the activation link sent by OMDb. "
                    "Note: It can take a few minutes for OMDb to activate newly registered keys."
                ),
                "movie": None,
                "plot_sentiment": None
            }
        elif resp.status_code != 200:
            return {
                "status": "error",
                "message": f"OMDb service returned status code {resp.status_code}.",
                "movie": None,
                "plot_sentiment": None
            }

        data = resp.json()
        if data.get("Response") == "False":
            return {
                "status": "not_found",
                "message": data.get("Error", f"No movie found matching '{title_clean}'."),
                "movie": None,
                "plot_sentiment": None
            }

        # Analyze plot sentiment if available
        plot_sentiment = None
        plot_text = data.get("Plot", "")
        if plot_text and plot_text != "N/A":
            try:
                plot_sentiment = predict_sentiment(plot_text, confidence_threshold=CONFIDENCE_THRESHOLD_ENV)
            except Exception:
                plot_sentiment = None

        return {
            "status": "success",
            "message": "Movie retrieved successfully.",
            "movie": data,
            "plot_sentiment": plot_sentiment
        }

    except httpx.RequestError as exc:
        return {
            "status": "error",
            "message": f"Network error contacting OMDb API: {str(exc)}",
            "movie": None,
            "plot_sentiment": None
        }


@app.post("/chat", response_model=ChatResponse, tags=["AI Chatbot"])
async def chat_with_cinebot(request: ChatRequest):
    """
    Answers movie queries using Google Gemini API when configured.
    Never exposes raw API keys.
    """
    query = request.message.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chat message cannot be empty."
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if not api_key:
        return {
            "status": "unconfigured",
            "reply": (
                "Google Gemini API key is not configured. Please add GEMINI_API_KEY "
                "to your .env file to enable the CineBot AI Assistant. "
                "You can get a free key at https://aistudio.google.com/app/apikey"
            ),
            "model_used": "none"
        }

    # Construct conversation payload for Gemini REST API
    system_instruction = (
        "You are CineBot, an expert film critic, cinema scholar, and helpful movie assistant. "
        "You provide engaging, articulate, and accurate answers about films, directors, screenplay structures, "
        "cinematic history, and movie review sentiment analysis. Keep answers concise, informative, and formatted with markdown."
    )

    gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    # Build contents array
    contents = []
    # Include history turns if present
    for turn in request.history:
        role = "user" if turn.role.lower() in ["user", "human"] else "model"
        contents.append({
            "role": role,
            "parts": [{"text": turn.content}]
        })

    # Add current query
    contents.append({
        "role": "user",
        "parts": [{"text": query}]
    })

    payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 800
        }
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(gemini_endpoint, json=payload)

        # Fallback to gemini-1.5-flash if 2.0-flash is unavailable or returns 404
        if resp.status_code == 404:
            fallback_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(fallback_endpoint, json=payload)

        if resp.status_code != 200:
            error_data = resp.json().get("error", {})
            error_msg = error_data.get("message", f"Gemini API returned status {resp.status_code}")
            return {
                "status": "error",
                "reply": f"Gemini API error: {error_msg}. Please verify your API key and quota.",
                "model_used": "gemini-error"
            }

        response_json = resp.json()
        candidates = response_json.get("candidates", [])
        if not candidates:
            return {
                "status": "error",
                "reply": "No response generated by the model. Please rephrase your question.",
                "model_used": "gemini"
            }

        reply_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return {
            "status": "success",
            "reply": reply_text.strip() or "No response generated.",
            "model_used": "gemini-flash"
        }

    except httpx.RequestError as exc:
        return {
            "status": "error",
            "reply": f"Network error contacting Gemini API: {str(exc)}",
            "model_used": "none"
        }


@app.get("/metrics", tags=["Model Evaluation"])
def get_model_metrics():
    """Returns actual evaluation metrics calculated during training."""
    try:
        metrics = load_metrics()
        return {
            "status": "success",
            "metrics": metrics
        }
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading metrics: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
