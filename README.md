# 🎬 Movie Sentiment Analysis Using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning web application designed for real-time movie review sentiment classification, batch CSV analysis, Explainable AI word attribution, verified film metadata exploration via **OMDb API**, and cinematic Q&A via **Google Gemini API**.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture & Tech Stack](#-architecture--tech-stack)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Key Features](#-key-features)
- [Authentication (Login & Signup)](#-authentication-login--signup)
- [Project Directory Structure](#-project-directory-structure)
- [Local Setup & Execution (macOS)](#-local-setup--execution-macos)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Clone & Virtual Environment Setup](#2-clone--virtual-environment-setup)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Environment Variables Configuration](#4-environment-variables-configuration)
  - [5. Train the Model](#5-train-the-model)
  - [6. Start the FastAPI Backend](#6-start-the-fastapi-backend)
  - [7. Launch the Streamlit Frontend](#7-launch-the-streamlit-frontend)
- [API Documentation](#-api-documentation)
- [Running Automated Tests](#-running-automated-tests)
- [Cloud Deployment (Render)](#-cloud-deployment-render)
- [Academic / Project Presentation Notes](#-academic--project-presentation-notes)

---

## 🎯 Project Overview

In the streaming and entertainment industry, millions of audience reviews are generated daily across platforms like IMDb, Rotten Tomatoes, and Letterboxd. Manually reviewing this feedback is impossible. 

This project implements a complete, decoupled web application:
1. **Machine Learning Pipeline**: Trained on IMDb reviews using **TF-IDF Vectorization** (unigrams + bigrams) and **Logistic Regression** with calibrated class probabilities.
2. **FastAPI Backend**: High-performance asynchronous REST API handling inference, batch predictions, OMDb metadata fetching, and Gemini consultations.
3. **Streamlit Frontend**: A responsive, cinema-themed dark UI featuring visual probability gauges, keyword impact chips, batch CSV processing with distribution charts, and interactive confusion matrices.

> [!NOTE]
> **Honest Classification Policy**: The IMDb dataset is binary (`positive` / `negative`). The application does **not** fabricate a synthetic 3-class training model. Reviews whose confidence score falls below a configurable decision threshold (default **60%**) are categorized as **"Neutral / Uncertain"** as an explicit threshold decision, preventing overconfident false predictions.

---

## 🏗️ Architecture & Tech Stack

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                   │
│          (app.py - Cinema Dark Theme on Port 8501)     │
└───────────────┬────────────────────────▲───────────────┘
                │ HTTP Requests          │ JSON Responses
                ▼                        │
┌────────────────────────────────────────────────────────┐
│                    FastAPI Backend                     │
│          (main.py - RESTful API on Port 8000)          │
└───────┬────────────────────────┬───────────────┬───────┘
        │                        │               │
        ▼                        ▼               ▼
┌──────────────────┐    ┌─────────────────┐ ┌───────────────┐
│ ML Engine        │    │ OMDb API        │ │ Google Gemini │
│ (ml_service.py)  │    │ (Film Metadata) │ │ (AI Chatbot)  │
│ - TF-IDF (1, 2)  │    └─────────────────┘ └───────────────┘
│ - Logistic Reg.  │
│ - Explainable AI │
└──────────────────┘
```

- **Backend**: FastAPI, Uvicorn, Pydantic, HTTPX
- **Frontend**: Streamlit, Matplotlib, Seaborn, Pandas
- **Machine Learning**: Scikit-Learn, NumPy, Joblib, NLTK
- **External Integrations**: OMDb API, Google Gemini API

---

## 🧠 Machine Learning Pipeline

1. **Text Preprocessing (`clean_text`)**:
   - Strips HTML tags (`<br />`) and web URLs.
   - Converts text to lowercase and strips non-alphabetic noise.
   - Preserves sentiment negation words (`not`, `no`, `never`, `hardly`).
   - Tokenizes and filters non-sentiment English stopwords.
2. **Train/Test Stratification**:
   - Uses an **80/20 Stratified Split** (`random_state=42`) ensuring identical class distributions in both training and test splits.
3. **Feature Extraction (`TfidfVectorizer`)**:
   - Computes Term Frequency-Inverse Document Frequency using unigrams and bigrams (`ngram_range=(1, 2)`).
   - Applies sublinear term frequency scaling (`sublinear_tf=True`) to dampen the impact of very high word repetitions.
4. **Classification (`LogisticRegression`)**:
   - Calibrated sigmoid probability estimates with balanced class weighting.
5. **Feature Attribution (Explainable AI)**:
   - Evaluates $\text{Weight}(w) = \text{Coefficient}(w) \times \text{TF-IDF}(w)$ for each token to quantify positive vs negative influence.

---

## ✨ Key Features

- 🔐 **Secure Login & Signup**: Cinema-themed first page with password strength meter; the dashboard and every data API require an authenticated session.
- 🎯 **Real-Time Sentiment Prediction**: Instant classification with confidence percentages.
- 🔍 **Explainable AI (Word Attribution)**: Highlights key terms driving positive (green) and negative (red) scores.
- ⚙️ **Configurable Uncertainty Threshold**: Dynamic slider (50%–90%) in the sidebar to control neutral/uncertain sensitivity.
- ⚡ **1-Click Test Presets**: Instant buttons for Masterpiece, Flop, Sci-Fi, and Nuanced reviews.
- 📜 **Session History**: Persists past predictions during the session with a clear history button.
- 📂 **Batch CSV Processing**: Upload CSV files, view live progress, inspect distribution charts, and download annotated CSVs.
- 🍿 **OMDb Movie Explorer**: Search any movie title to inspect verified cast, director, awards, and run sentiment analysis on the storyline plot.
- 🤖 **CineBot AI Assistant**: Chat with an AI cinema critic powered by Google Gemini.
- 📊 **Model Metrics & Confusion Matrix**: Real, un-fabricated confusion matrix heatmaps and classification reports.

---

## 🔐 Authentication (Login & Signup)

Anonymous visitors only ever see the **Login / Signup** page. After logging in they are taken to the dashboard, which has a **Logout** button in its header.

| Concern | How it works |
| :--- | :--- |
| Password storage | Hashed with **bcrypt** (salted, adaptive cost). Plain-text passwords are never stored, logged, or returned by any API. |
| Password policy | 8+ characters with an uppercase letter, a lowercase letter and a number (max 72 bytes, a bcrypt limit). The signup form shows a live strength meter; the backend enforces the same rules (`auth_rules.py`). |
| Sessions | Signed **JWT bearer tokens** (default lifetime 60 min), kept in Streamlit's server-side session state, never in the URL. |
| Logout | Clears the Streamlit session **and revokes the token on the server**, so it stops working immediately. |
| Protected APIs | `/predict`, `/predict/batch`, `/movie/{title}`, `/movies/catalog`, `/metrics` and `/auth/me` return **401** without a valid token. `/`, `/health`, `/auth/signup` and `/auth/login` are public. |
| Brute force | After 5 failed logins for an account (per server process), further attempts get **429** for 15 minutes. Login errors never reveal whether an email exists. |
| Browser refresh | A refresh starts a new Streamlit session, so you are logged out and must log in again. A refresh can never skip authentication. |
| Secrets | `JWT_SECRET_KEY` and `DATABASE_URL` come from environment variables only. `.env` and `*.db` are git-ignored. |

**User database:** SQLAlchemy models in `database.py` (`users`, `revoked_tokens`). Locally it defaults to a SQLite file (`users.db`, created automatically). For deployment set `DATABASE_URL` to a **PostgreSQL** URL, because free hosts such as Render erase local files on restart, which would delete SQLite accounts.

---

## 📁 Project Directory Structure

```text
movie-sentiment-app/
├── app.py                      # Streamlit frontend application (dashboard)
├── auth_ui.py                  # Streamlit Login / Signup page, session handling, Logout
├── auth_rules.py               # Password / email rules shared by frontend and backend
├── auth.py                     # Backend auth: bcrypt hashing, JWT tokens, /auth/* routes
├── database.py                 # SQLAlchemy engine and User / RevokedToken models
├── main.py                     # FastAPI REST API backend
├── ml_service.py               # Shared ML inference and explainability engine
├── train_model.py              # Model training & metrics generation pipeline
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git exclusion rules
├── README.md                   # Complete documentation
├── data/
│   ├── imdb_dataset.csv        # Curated benchmark dataset (300 diverse reviews)
│   └── README.md               # Instructions for Kaggle 50K IMDB dataset
├── models/
│   ├── sentiment_model.joblib  # Serialized pipeline (TF-IDF + Logistic Regression)
│   └── metrics.json            # Genuine test set evaluation metrics
└── tests/
    ├── __init__.py
    ├── test_api.py             # FastAPI integration tests (authenticated)
    ├── test_auth.py            # Signup / login / logout / protected-route / security tests
    ├── test_model.py           # ML inference and threshold tests
    └── test_dataset.py         # Dataset validation & edge cases tests
```

---

## 🚀 Local Setup & Execution (macOS)

### 1. Prerequisites
- macOS with **Python 3.10+** installed
- Terminal access

### 2. Clone & Virtual Environment Setup
Open your terminal and navigate to the project directory:

```bash
cd /Users/suraj/.gemini/antigravity/scratch/movie-sentiment-app
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment Variables Configuration
Copy the template configuration file:

```bash
cp .env.example .env
```

Edit `.env` to add your optional API keys (the app works fully without them for sentiment analysis):
- **OMDb API Key**: Get a free key at [https://www.omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx)
- **Google Gemini API Key**: Get a free key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

```ini
FASTAPI_URL=http://127.0.0.1:8000
CONFIDENCE_THRESHOLD=0.60
OMDB_API_KEY=your_omdb_key_here
GEMINI_API_KEY=your_gemini_key_here

# Authentication
JWT_SECRET_KEY=paste_a_long_random_value_here
DATABASE_URL=
```

Generate a signing key for `JWT_SECRET_KEY` (never commit it):

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

| Variable | Required | Purpose |
| :--- | :--- | :--- |
| `JWT_SECRET_KEY` | **Yes (production)** | Signs login tokens. If unset, a temporary random key is used and all users are logged out on every backend restart. |
| `DATABASE_URL` | Production | User database. Empty = local SQLite `users.db`. Use `postgresql://user:pass@host:5432/db` in production. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | Session lifetime (default `60`). |
| `LOGIN_MAX_ATTEMPTS` / `LOGIN_LOCKOUT_SECONDS` | No | Failed-login throttle (default `5` / `900`). |
| `CORS_ORIGINS` | No | Comma-separated allowed browser origins (default all). |

### 5. Train the Model
Run the model training pipeline:

```bash
python3 train_model.py
```

This will preprocess the dataset, perform an 80/20 stratified split, evaluate on the holdout test set, and save:
- Model pipeline: `models/sentiment_model.joblib`
- Metrics: `models/metrics.json`

*(Optional: To train on Kaggle's 50K IMDB dataset, place `IMDB Dataset.csv` inside `data/` and run `python3 train_model.py`)*

### 6. Start the FastAPI Backend
In your first terminal tab:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Verify backend is running by opening:
- API Status: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Swagger Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 7. Launch the Streamlit Frontend
In a **second terminal tab** (with the virtual environment activated):

```bash
cd /Users/suraj/.gemini/antigravity/scratch/movie-sentiment-app
source venv/bin/activate
streamlit run app.py --server.port 8501
```

Open your browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

The first page is **Login / Signup**. Create an account (the first time), log in, and you will land on the dashboard.

---

## 📡 API Documentation

FastAPI provides an automatic, interactive Swagger UI at **`/docs`**.

🔒 = requires `Authorization: Bearer <token>` (get one from `/auth/login`); otherwise the API returns `401`.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and endpoint directory |
| `GET` | `/health` | Health check (model readiness & external API status) |
| `POST` | `/auth/signup` | Create an account (`full_name`, `email`, `password`, `confirm_password`) |
| `POST` | `/auth/login` | Log in (`email`, `password`); returns a bearer token |
| `POST` | `/auth/logout` 🔒 | End the session (revokes the token) |
| `GET` | `/auth/me` 🔒 | Current user |
| `POST` | `/predict` 🔒 | Predict sentiment, confidence, and keyword attribution |
| `POST` | `/predict/batch` 🔒 | Bulk classification for an array of reviews |
| `GET` | `/movie/{title}` 🔒 | Fetch OMDb movie metadata & analyze plot sentiment |
| `GET` | `/movies/catalog` 🔒 | List the built-in movie catalog |
| `POST` | `/chat` | Chat with CineBot powered by Google Gemini |
| `GET` | `/metrics` 🔒 | Actual model test evaluation metrics from `metrics.json` |

Example:

```bash
curl -X POST http://127.0.0.1:8000/auth/login -H "Content-Type: application/json" \
     -d '{"email": "you@example.com", "password": "YourPassword1"}'
curl -X POST http://127.0.0.1:8000/predict -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" -d '{"review": "A brilliant film!"}'
```

---

## 🧪 Running Automated Tests

Run the full automated test suite covering ML inference, edge cases, FastAPI endpoints, and dataset validation:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Or using pytest:

```bash
pytest tests/ -v
```

Tests use an isolated temporary database and never touch your real `users.db`. They verify:
- Authentication: signup, duplicate-email rejection, login, wrong passwords, protected endpoints (401 without a token, forged/expired tokens), logout revocation, bcrypt hashing, no passwords in responses or logs
- Positive, negative, and threshold-based uncertain classifications
- Empty and invalid review handling (400 Bad Request)
- Batch prediction accuracy and edge cases
- Graceful fallbacks when external API keys are unconfigured
- Model file missing exceptions

---

## ☁️ Cloud Deployment (Render)

### Deploying the FastAPI Backend on Render
1. Create a **New Web Service** on [Render](https://render.com).
2. Set Build Command:
   ```bash
   pip install -r requirements.txt && python train_model.py
   ```
3. Set Start Command:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
4. Add Environment Variables in the Render dashboard:
   - `CONFIDENCE_THRESHOLD`: `0.60`
   - `OMDB_API_KEY`: *(your key)*
   - `GEMINI_API_KEY`: *(your key)*
   - `JWT_SECRET_KEY`: *(long random value; `render.yaml` can generate one automatically)*
   - `DATABASE_URL`: *(PostgreSQL connection string, e.g. from Neon, Supabase or Render Postgres)*

> [!IMPORTANT]
> Without `DATABASE_URL` the backend falls back to a local SQLite file. Render's free tier has an **ephemeral disk**, so every account would be erased when the service restarts or wakes from sleep. Use a managed PostgreSQL database for real deployments. Accounts are created per backend database: after deploying, sign up again on the live app.

### Deploying the Streamlit Frontend on Streamlit Community Cloud
1. Connect your GitHub repository to [Streamlit Community Cloud](https://share.streamlit.io).
2. Set Main file path: `app.py`.
3. Set Secrets:
   ```toml
   FASTAPI_URL = "https://your-fastapi-backend.onrender.com"
   ```

---

## 🎓 Academic / Project Presentation Notes

When presenting this project for an academic viva or evaluation:
1. **Explain the Model Choice**: Logistic Regression was chosen over black-box deep learning models because it offers fast inference, well-calibrated posterior probabilities, and direct explainability via feature weights.
2. **Explain TF-IDF**: Highlights how rare, highly informative sentiment terms (e.g. *"breathtaking"*, *"cringeworthy"*) receive higher mathematical weights than generic words.
3. **Highlight Honest Binary Thresholding**: Point out that the model does not pretend to have been trained on a neutral class. Reviews close to 50/50 probability are labeled Neutral/Uncertain via decision thresholding to prevent false confidence.
4. **Demonstrate Decoupled Architecture**: Explain the software engineering advantage of separating the FastAPI microservice from the Streamlit presentation layer.
