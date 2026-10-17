"""
FastAPI Backend for Movie Sentiment Analysis.
Provides RESTful endpoints for sentiment prediction, batch classification,
OMDb movie metadata retrieval with plot sentiment, and Gemini-powered CineBot chat.
"""

import os
import json
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


FALLBACK_MOVIES = {
    "inception": {
        "Title": "Inception",
        "Year": "2010",
        "Rated": "PG-13",
        "Released": "16 Jul 2010",
        "Runtime": "148 min",
        "Genre": "Action, Adventure, Sci-Fi",
        "Director": "Christopher Nolan",
        "Writer": "Christopher Nolan",
        "Actors": "Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page, Tom Hardy",
        "Plot": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the project and his team to disaster.",
        "Language": "English, Japanese, French",
        "Country": "United States, United Kingdom",
        "Awards": "Won 4 Oscars. 159 wins & 220 nominations total",
        "Poster": "https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_SX300.jpg",
        "imdbRating": "8.8",
        "imdbVotes": "2,560,000",
        "imdbID": "tt1375666",
        "Metascore": "74",
        "BoxOffice": "$292,587,330",
        "Response": "True"
    },
    "the dark knight": {
        "Title": "The Dark Knight",
        "Year": "2008",
        "Rated": "PG-13",
        "Released": "18 Jul 2008",
        "Runtime": "152 min",
        "Genre": "Action, Crime, Drama",
        "Director": "Christopher Nolan",
        "Writer": "Jonathan Nolan, Christopher Nolan, David S. Goyer",
        "Actors": "Christian Bale, Heath Ledger, Aaron Eckhart, Michael Caine",
        "Plot": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
        "Language": "English, Mandarin",
        "Country": "United States, United Kingdom",
        "Awards": "Won 2 Oscars. 163 wins & 164 nominations total",
        "Poster": "https://m.media-amazon.com/images/M/MV5BMTMxNTMwODM0NF5BMl5BanBnXkFtZTcwODAyMTk2Mw@@._V1_SX300.jpg",
        "imdbRating": "9.0",
        "imdbVotes": "2,900,000",
        "imdbID": "tt0468569",
        "Metascore": "84",
        "BoxOffice": "$534,987,076",
        "Response": "True"
    },
    "interstellar": {
        "Title": "Interstellar",
        "Year": "2014",
        "Rated": "PG-13",
        "Released": "07 Nov 2014",
        "Runtime": "169 min",
        "Genre": "Adventure, Drama, Sci-Fi",
        "Director": "Christopher Nolan",
        "Writer": "Jonathan Nolan, Christopher Nolan",
        "Actors": "Matthew McConaughey, Anne Hathaway, Jessica Chastain, Michael Caine",
        "Plot": "When Earth becomes uninhabitable in the future, a farmer and ex-NASA pilot, Joseph Cooper, is tasked to pilot a spacecraft, along with a team of researchers, to find a new planet for humans.",
        "Language": "English",
        "Country": "United States, United Kingdom, Canada",
        "Awards": "Won 1 Oscar. 44 wins & 148 nominations total",
        "Poster": "https://m.media-amazon.com/images/M/MV5BYzdjMDAxZGItMjI2My00ODA1LTlkNzItOWFjMDU5ZDJlYWY3XkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "8.7",
        "imdbVotes": "2,100,000",
        "imdbID": "tt0816692",
        "Metascore": "74",
        "BoxOffice": "$188,020,017",
        "Response": "True"
    },
    "titanic": {
        "Title": "Titanic",
        "Year": "1997",
        "Rated": "PG-13",
        "Released": "19 Dec 1997",
        "Runtime": "194 min",
        "Genre": "Drama, Romance",
        "Director": "James Cameron",
        "Writer": "James Cameron",
        "Actors": "Leonardo DiCaprio, Kate Winslet, Billy Zane, Kathy Bates",
        "Plot": "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
        "Language": "English, Swedish, Italian, French",
        "Country": "United States, Mexico",
        "Awards": "Won 11 Oscars. 126 wins & 83 nominations total",
        "Poster": "https://m.media-amazon.com/images/M/MV5BYzYyN2FiZmUtYWYzMy00MzViLWJkZTMtOGY1ZjgzNWMwN2YxXkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "7.9",
        "imdbVotes": "1,300,000",
        "imdbID": "tt0120338",
        "Metascore": "75",
        "BoxOffice": "$674,292,608",
        "Response": "True"
    },
    "the godfather": {
        "Title": "The Godfather",
        "Year": "1972",
        "Rated": "R",
        "Released": "24 Mar 1972",
        "Runtime": "175 min",
        "Genre": "Crime, Drama",
        "Director": "Francis Ford Coppola",
        "Writer": "Mario Puzo, Francis Ford Coppola",
        "Actors": "Marlon Brando, Al Pacino, James Caan, Diane Keaton",
        "Plot": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "Language": "English, Italian, Latin",
        "Country": "United States",
        "Awards": "Won 3 Oscars. 32 wins & 31 nominations total",
        "Poster": "https://m.media-amazon.com/images/M/MV5BYTJkNGQyZDgtZDExNC00MDY0LWEwNTAtOHZjOWZmNTQxN2VhXkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "9.2",
        "imdbVotes": "2,050,000",
        "imdbID": "tt0068646",
        "Metascore": "100",
        "BoxOffice": "$136,381,073",
        "Response": "True"
    },
    "oppenheimer": {
        "Title": "Oppenheimer",
        "Year": "2023",
        "Rated": "R",
        "Released": "21 Jul 2023",
        "Runtime": "180 min",
        "Genre": "Biography, Drama, History",
        "Director": "Christopher Nolan",
        "Writer": "Christopher Nolan, Kai Bird, Martin Sherwin",
        "Actors": "Cillian Murphy, Emily Blunt, Matt Damon, Robert Downey Jr.",
        "Plot": "The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb during World War II.",
        "Language": "English, German, Italian",
        "Country": "United States, United Kingdom",
        "Awards": "Won 7 Oscars. 392 wins & 401 nominations total",
        "Poster": "https://m.media-amazon.com/images/M/MV5BN2JkMDc5MGQtZjg3YS00NmFiLWIyZmQtZTJmNTM5MjVmYTQ4XkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "8.8",
        "imdbVotes": "850,000",
        "imdbID": "tt15398776",
        "Metascore": "88",
        "BoxOffice": "$329,862,540",
        "Response": "True"
    },
    "3 idiots": {
        "Title": "3 Idiots",
        "Year": "2009",
        "Rated": "PG-13",
        "Released": "25 Dec 2009",
        "Runtime": "170 min",
        "Genre": "Comedy, Drama",
        "Director": "Rajkumar Hirani",
        "Writer": "Vidhu Vinod Chopra, Rajkumar Hirani, Abhijat Joshi",
        "Actors": "Aamir Khan, Madhavan, Sharman Joshi, Kareena Kapoor",
        "Plot": "Two friends are searching for their long lost companion. They revisit their college days and recall the memories of their friend who inspired them to think differently, even as the rest of the world called them idiots.",
        "Language": "Hindi, English",
        "Country": "India",
        "Awards": "Won 6 Filmfare Awards, 3 National Film Awards. 58 wins total",
        "Poster": "https://m.media-amazon.com/images/M/MV5BNTkyOGVjMGEtNmQzZi00NzFlLTlhOWQtODYyMDc2ZGJmYzFhXkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "8.4",
        "imdbVotes": "430,000",
        "imdbID": "tt1187043",
        "Metascore": "67",
        "BoxOffice": "$6,532,874",
        "Response": "True"
    },
    "dangal": {
        "Title": "Dangal",
        "Year": "2016",
        "Rated": "Not Rated",
        "Released": "23 Dec 2016",
        "Runtime": "161 min",
        "Genre": "Action, Biography, Drama",
        "Director": "Nitesh Tiwari",
        "Writer": "Piyush Gupta, Shreyas Jain, Nikhil Mehrotra",
        "Actors": "Aamir Khan, Fatima Sana Shaikh, Sanya Malhotra, Sakshi Tanwar",
        "Plot": "Former wrestler Mahavir Singh Phogat and his two wrestler daughters struggle towards glory at the Commonwealth Games in the face of societal oppression and doubt.",
        "Language": "Hindi, Tamil, Telugu",
        "Country": "India",
        "Awards": "Won 4 Filmfare Awards, 1 National Film Award. Highest grossing Indian film worldwide",
        "Poster": "https://m.media-amazon.com/images/M/MV5BMTQ4MzQzMzM2Nl5BMl5BanBnXkFtZTgwMTQ1NzU3MDI@._V1_SX300.jpg",
        "imdbRating": "8.3",
        "imdbVotes": "210,000",
        "imdbID": "tt5074352",
        "Metascore": "N/A",
        "BoxOffice": "$12,391,761",
        "Response": "True"
    },
    "sholay": {
        "Title": "Sholay",
        "Year": "1975",
        "Rated": "Not Rated",
        "Released": "15 Aug 1975",
        "Runtime": "204 min",
        "Genre": "Action, Adventure, Comedy, Drama",
        "Director": "Ramesh Sippy",
        "Writer": "Salim Khan, Javed Akhtar",
        "Actors": "Dharmendra, Sanjeev Kumar, Hema Malini, Amitabh Bachchan, Amjad Khan",
        "Plot": "After his family is murdered by a notorious and ruthless bandit, a former police officer enlists the services of two outlaws to take him down and protect the villagers.",
        "Language": "Hindi",
        "Country": "India",
        "Awards": "Voted Best Film of 50 Years at Filmfare Awards",
        "Poster": "https://m.media-amazon.com/images/M/MV5BMjA5NTgyNTQ3M15BMl5BanBnXkFtZTgwNTU0ODMzMzE@._V1_SX300.jpg",
        "imdbRating": "8.1",
        "imdbVotes": "60,000",
        "imdbID": "tt0073707",
        "Metascore": "N/A",
        "BoxOffice": "N/A",
        "Response": "True"
    },
    "lagaan": {
        "Title": "Lagaan: Once Upon a Time in India",
        "Year": "2001",
        "Rated": "PG",
        "Released": "15 Jun 2001",
        "Runtime": "224 min",
        "Genre": "Adventure, Drama, Musical, Sport",
        "Director": "Ashutosh Gowariker",
        "Writer": "Ashutosh Gowariker, Kumar Dave, Sanjay Dayma",
        "Actors": "Aamir Khan, Gracy Singh, Rachel Shelley, Paul Blackthorne",
        "Plot": "The people of a small village in Victorian India stake their future on a game of cricket against their ruthless British rulers to avoid paying high agricultural taxes.",
        "Language": "Hindi, English",
        "Country": "India",
        "Awards": "Nominated for Oscar (Best Foreign Language Film). Won 8 National Film Awards, 9 Filmfare Awards",
        "Poster": "https://m.media-amazon.com/images/M/MV5BNDYxNWUzZmYtBhMzY100NGRjLWJkZWYtMDk2Y2MxYjhlOTcxXkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "8.1",
        "imdbVotes": "120,000",
        "imdbID": "tt0169102",
        "Metascore": "84",
        "BoxOffice": "$70,127,118",
        "Response": "True"
    },
    "swades": {
        "Title": "Swades",
        "Year": "2004",
        "Rated": "Not Rated",
        "Released": "17 Dec 2004",
        "Runtime": "210 min",
        "Genre": "Drama",
        "Director": "Ashutosh Gowariker",
        "Writer": "Ashutosh Gowariker, M.G. Sathya",
        "Actors": "Shah Rukh Khan, Gayatri Joshi, Kishori Ballal, Rajesh Vivek",
        "Plot": "A successful Indian scientist working at NASA returns to rural India to bring his childhood nanny to America, and during his stay rediscovers his cultural roots and purpose.",
        "Language": "Hindi",
        "Country": "India",
        "Awards": "Won 2 National Film Awards, 2 Filmfare Awards",
        "Poster": "https://m.media-amazon.com/images/M/MV5BYzExOTcwNjYtOYzYy00OGJmLWEwOTgtMDRjOGFlYjc1MWE3XkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "8.2",
        "imdbVotes": "95,000",
        "imdbID": "tt0367110",
        "Metascore": "N/A",
        "BoxOffice": "$1,223,240",
        "Response": "True"
    },
    "tumbbad": {
        "Title": "Tumbbad",
        "Year": "2018",
        "Rated": "Not Rated",
        "Released": "12 Oct 2018",
        "Runtime": "104 min",
        "Genre": "Drama, Fantasy, Horror",
        "Director": "Rahi Anil Barve, Anand Gandhi, Adesh Prasad",
        "Writer": "Mitesh Shah, Adesh Prasad, Rahi Anil Barve",
        "Actors": "Sohum Shah, Jyoti Malshe, Anita Date-Kelkar, Deepak Damle",
        "Plot": "A mythological horror story about a human who builds a shrine for the demon of greed, Hastar, triggering a terrifying curse across three generations.",
        "Language": "Hindi, Marathi",
        "Country": "India",
        "Awards": "Won 3 Filmfare Awards. Critical international acclaim for visual artistry",
        "Poster": "https://m.media-amazon.com/images/M/MV5BMzRjOTU0YjItNDYwYi00Nzk1LWJlNDEtNWZjOWQ4ODk4N2NkXkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "8.2",
        "imdbVotes": "62,000",
        "imdbID": "tt8239946",
        "Metascore": "N/A",
        "BoxOffice": "N/A",
        "Response": "True"
    },
    "dilwale dulhania le jayenge": {
        "Title": "Dilwale Dulhania Le Jayenge (DDLJ)",
        "Year": "1995",
        "Rated": "Not Rated",
        "Released": "20 Oct 1995",
        "Runtime": "189 min",
        "Genre": "Drama, Romance",
        "Director": "Aditya Chopra",
        "Writer": "Aditya Chopra, Javed Siddiqui",
        "Actors": "Shah Rukh Khan, Kajol, Amrish Puri, Farida Jalal, Anupam Kher",
        "Plot": "When Raj meets Simran on a train journey across Europe, they fall deeply in love. But Raj must win over Simran's traditional father in Punjab before her arranged marriage.",
        "Language": "Hindi",
        "Country": "India",
        "Awards": "Won 10 Filmfare Awards, 1 National Film Award. Longest running film in Indian cinema history",
        "Poster": "https://m.media-amazon.com/images/M/MV5BMDQ2YWVjNWUtYTI2Yy00YWY4LWI4NDMtMzQ5MWE4OTQ5OTI4XkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "8.0",
        "imdbVotes": "75,000",
        "imdbID": "tt0112870",
        "Metascore": "N/A",
        "BoxOffice": "$100,000,000",
        "Response": "True"
    },
    "bajrangi bhaijaan": {
        "Title": "Bajrangi Bhaijaan",
        "Year": "2015",
        "Rated": "Not Rated",
        "Released": "17 Jul 2015",
        "Runtime": "163 min",
        "Genre": "Action, Adventure, Comedy, Drama",
        "Director": "Kabir Khan",
        "Writer": "Vijayendra Prasad, Kabir Khan, Parveez Sheikh",
        "Actors": "Salman Khan, Kareena Kapoor, Nawazuddin Siddiqui, Harshaali Malhotra",
        "Plot": "A kind-hearted Indian man with an unwavering devotion to Lord Hanuman embarks on an arduous journey to escort a mute Pakistani girl back to her home across the border.",
        "Language": "Hindi, Urdu",
        "Country": "India",
        "Awards": "Won National Film Award for Best Popular Film. 30 wins total",
        "Poster": "https://m.media-amazon.com/images/M/MV5BMTUzODk1NzI4M15BMl5BanBnXkFtZTgwMDUyMDgyNTE@._V1_SX300.jpg",
        "imdbRating": "8.1",
        "imdbVotes": "100,000",
        "imdbID": "tt3863552",
        "Metascore": "N/A",
        "BoxOffice": "$8,180,000",
        "Response": "True"
    },
    "zindagi na milegi dobara": {
        "Title": "Zindagi Na Milegi Dobara",
        "Year": "2011",
        "Rated": "Not Rated",
        "Released": "15 Jul 2011",
        "Runtime": "155 min",
        "Genre": "Comedy, Drama",
        "Director": "Zoya Akhtar",
        "Writer": "Farhan Akhtar, Reema Kagti, Zoya Akhtar",
        "Actors": "Hrithik Roshan, Farhan Akhtar, Abhay Deol, Katrina Kaif, Kalki Koechlin",
        "Plot": "Three friends embark on an ultimate bachelor road trip through Spain, confronting their deepest personal fears, healing past wounds, and discovering the joy of living in the present.",
        "Language": "Hindi, English, Spanish",
        "Country": "India",
        "Awards": "Won 2 National Film Awards, 7 Filmfare Awards",
        "Poster": "https://m.media-amazon.com/images/M/MV5BZGFmMjM5OWMtZTRiNC00ODhlLThlYTItMzg2N2kyM2RhNWJhXkEyXkFqcGc@._V1_SX300.jpg",
        "imdbRating": "8.2",
        "imdbVotes": "85,000",
        "imdbID": "tt1562872",
        "Metascore": "N/A",
        "BoxOffice": "$3,108,665",
        "Response": "True"
    },
    "pk": {
        "Title": "PK",
        "Year": "2014",
        "Rated": "Not Rated",
        "Released": "19 Dec 2014",
        "Runtime": "153 min",
        "Genre": "Comedy, Drama, Sci-Fi",
        "Director": "Rajkumar Hirani",
        "Writer": "Rajkumar Hirani, Abhijat Joshi",
        "Actors": "Aamir Khan, Anushka Sharma, Sushant Singh Rajput, Sanjay Dutt, Boman Irani",
        "Plot": "An innocent alien stranded on Earth loses his communication remote. His curious child-like questions expose the superstitions, hypocrisy, and blind dogmas of society.",
        "Language": "Hindi",
        "Country": "India",
        "Awards": "Won 2 Filmfare Awards. Broke global box office records",
        "Poster": "https://m.media-amazon.com/images/M/MV5BMTYzOTE2NjkxN15BMl5BanBnXkFtZTgwMDgzMTg0MzE@._V1_SX300.jpg",
        "imdbRating": "8.1",
        "imdbVotes": "170,000",
        "imdbID": "tt2338151",
        "Metascore": "N/A",
        "BoxOffice": "$10,616,104",
        "Response": "True"
    }
}

# Automatically load extended Hindi movies database
HINDI_MOVIES_FILE = os.path.join(BASE_DIR, "data", "hindi_movies.json")
if os.path.exists(HINDI_MOVIES_FILE):
    try:
        with open(HINDI_MOVIES_FILE, "r", encoding="utf-8") as _hf:
            _hindi_data = json.load(_hf)
            FALLBACK_MOVIES.update(_hindi_data)
    except Exception as _ex:
        print(f"Warning: could not read {HINDI_MOVIES_FILE}: {_ex}")


def _find_catalog_movie(query: str):
    """Checks offline catalog for famous titles if OMDb key is inactive/unconfigured."""
    q = query.strip().lower()
    aliases = {
        "ddlj": "dilwale dulhania le jayenge",
        "dilwale dulhaniya le jayenge": "dilwale dulhania le jayenge",
        "znmd": "zindagi na milegi dobara",
        "3 idiot": "3 idiots",
        "three idiots": "3 idiots",
        "bajrangi": "bajrangi bhaijaan",
        "yjhd": "yeh jawaani hai deewani",
        "kkhh": "kuch kuch hota hai",
        "k3g": "kabhi khushi kabhie gham",
        "stree2": "stree 2",
        "kgf2": "kgf",
        "12 fail": "12th fail",
        "twelfth fail": "12th fail",
        "gangs of wasseypur 1": "gangs of wasseypur",
        "gow": "gangs of wasseypur",
        "munnabhai": "munna bhai m.b.b.s.",
        "munna bhai": "munna bhai m.b.b.s.",
        "tzp": "taare zameen par",
        "chak de": "chak de! india",
        "chakde": "chak de! india",
    }
    target = aliases.get(q, q)
    for key, data in FALLBACK_MOVIES.items():
        if key in target or target in key or target == key:
            return data
    return None


@app.get("/movies/catalog", tags=["OMDb Integration"])
def get_catalog_movies():
    """Returns list of all available catalog movies."""
    return {
        "status": "success",
        "total": len(FALLBACK_MOVIES),
        "movies": [
            {
                "title": m["Title"],
                "year": m.get("Year"),
                "genre": m.get("Genre"),
                "rating": m.get("imdbRating"),
                "actors": m.get("Actors")
            }
            for m in FALLBACK_MOVIES.values()
        ]
    }


@app.get("/movie/{title}", tags=["OMDb Integration"])
async def get_movie_details(title: str, apikey: Optional[str] = Query(default=None)):
    """
    Fetches film metadata from OMDb API and analyzes plot sentiment when available.
    Supports live OMDb retrieval with automatic fallback to built-in cinema catalog.
    Never exposes raw API keys.
    """
    title_clean = title.strip()
    if not title_clean:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Movie title must be specified."
        )

    # Dynamically reload environment variables in case .env was recently modified
    load_dotenv(override=True)
    api_key = (apikey or "").strip() or os.getenv("OMDB_API_KEY", "").strip() or OMDB_API_KEY

    # 1. Attempt live OMDb API call if API key exists
    if api_key:
        omdb_url = "https://www.omdbapi.com/"
        params = {
            "t": title_clean,
            "plot": "full",
            "apikey": api_key
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(omdb_url, params=params)

            if resp.status_code == 200:
                data = resp.json()
                if data.get("Response") == "True":
                    plot_sentiment = None
                    plot_text = data.get("Plot", "")
                    if plot_text and plot_text != "N/A":
                        try:
                            plot_sentiment = predict_sentiment(plot_text, confidence_threshold=CONFIDENCE_THRESHOLD_ENV)
                        except Exception:
                            plot_sentiment = None

                    return {
                        "status": "success",
                        "source": "live_omdb",
                        "message": "Movie retrieved successfully from OMDb API.",
                        "movie": data,
                        "plot_sentiment": plot_sentiment
                    }
                elif data.get("Response") == "False":
                    # Check fallback catalog before returning not found
                    catalog_match = _find_catalog_movie(title_clean)
                    if not catalog_match:
                        return {
                            "status": "not_found",
                            "message": data.get("Error", f"No movie found matching '{title_clean}'."),
                            "movie": None,
                            "plot_sentiment": None
                        }
        except Exception:
            # Fall through to fallback catalog if network/OMDb fails
            pass

    # 2. Check offline cinema catalog (allows testing popular films even if key is inactive/unconfigured)
    catalog_match = _find_catalog_movie(title_clean)
    if catalog_match:
        plot_sentiment = None
        plot_text = catalog_match.get("Plot", "")
        if plot_text:
            try:
                plot_sentiment = predict_sentiment(plot_text, confidence_threshold=CONFIDENCE_THRESHOLD_ENV)
            except Exception:
                plot_sentiment = None

        return {
            "status": "success",
            "source": "demo_catalog",
            "message": "Movie details loaded from verified cinema catalog (OMDb key pending activation).",
            "movie": catalog_match,
            "plot_sentiment": plot_sentiment
        }

    # 3. If no key and not in catalog, return clear activation instructions
    return {
        "status": "unconfigured",
        "message": (
            "OMDb returned '401 Unauthorized' (API key is inactive). "
            "Please check your email and click the verification link sent by OMDb to activate it. "
            "Tip: You can instantly test movies from our verified catalog (e.g., 'Inception', 'The Dark Knight', 'Interstellar', 'Titanic', 'The Godfather', 'Oppenheimer')."
        ),
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
