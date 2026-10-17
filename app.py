"""
Streamlit Frontend for Movie Sentiment Analysis.
Features:
- Dark Cinema UI theme
- Single review analyzer with Explainable AI (word attribution)
- Configurable threshold for Neutral / Uncertain decision
- Quick preset sample buttons
- Session state prediction history
- Batch CSV upload, analysis, and download
- OMDb movie lookup with plot narrative sentiment analysis
- Model performance metrics & confusion matrix visualization
"""

import os
import io
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="CinePulse — Movie Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration defaults
DEFAULT_BACKEND_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")

# Custom Cinema Dark Theme CSS
CINEMA_CSS = """
<style>
    /* Dark Cinema Color Palette */
    :root {
        --cinema-bg: #0b0f19;
        --card-bg: #131b2e;
        --card-border: #1e293b;
        --gold-accent: #f5c518;
        --red-accent: #e50914;
        --green-accent: #00e676;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    /* Main body styling */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #131b2e 0%, #1a243d 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Sentiment Badges */
    .badge-positive {
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid #00e676;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-negative {
        background: rgba(229, 9, 20, 0.15);
        color: #ff5252;
        border: 1px solid #ff5252;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-neutral {
        background: rgba(245, 197, 24, 0.15);
        color: #f5c518;
        border: 1px solid #f5c518;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* Word attribution tags */
    .tag-pos {
        background: rgba(0, 230, 118, 0.2);
        color: #69f0ae;
        border: 1px solid #00e676;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.85rem;
        margin: 4px;
        display: inline-block;
    }
    .tag-neg {
        background: rgba(229, 9, 20, 0.2);
        color: #ff8a80;
        border: 1px solid #ff5252;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.85rem;
        margin: 4px;
        display: inline-block;
    }

    /* Dual gauge bar */
    .gauge-wrapper {
        background: #1e293b;
        border-radius: 10px;
        height: 22px;
        width: 100%;
        overflow: hidden;
        display: flex;
        margin: 12px 0;
    }
    .gauge-pos {
        background: linear-gradient(90deg, #00c853, #00e676);
        height: 100%;
        transition: width 0.4s ease;
    }
    .gauge-neg {
        background: linear-gradient(90deg, #d50000, #ff5252);
        height: 100%;
        transition: width 0.4s ease;
    }

    /* Notice banner */
    .cinema-banner {
        background: #131b2e;
        border-left: 4px solid #f5c518;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 20px;
    }
</style>
"""
st.markdown(CINEMA_CSS, unsafe_allow_html=True)

# Curated sample reviews for quick 1-click testing (including top Hindi films)
SAMPLE_REVIEWS = [
    {
        "label": "🍿 3 Idiots (Masterpiece)",
        "text": "An absolute emotional and inspiring masterpiece! Brilliant performances by Aamir Khan, Sharman, and Madhavan. The humor was brilliant and the message touched my heart."
    },
    {
        "label": "🥊 Dangal (Sports Blockbuster)",
        "text": "Sensational biographical sports drama! Powerhouse performance by Aamir Khan, breathtaking wrestling choreography, and pure emotional goosebumps throughout."
    },
    {
        "label": "👻 Tumbbad (Mythological Horror)",
        "text": "A visual and mythological horror masterpiece of Indian cinema. Atmospheric, haunting, exquisitely shot, and an unforgettable story about greed."
    },
    {
        "label": "👎 Bollywood Flop",
        "text": "A complete and utter waste of time. Mindless script, cringeworthy dialogue, loud unnecessary songs, and wooden acting throughout."
    },
    {
        "label": "🤔 Mixed Hindi Review",
        "text": "The music and cinematography were reasonably good, though the screenplay had pacing issues and the second half felt overly melodramatic."
    }
]

# Initialize session states
if "history" not in st.session_state:
    st.session_state.history = []
if "review_input" not in st.session_state:
    st.session_state.review_input = ""


# ==========================================
# Sidebar: Backend Status & Settings
# ==========================================

st.sidebar.markdown("## 🎬 **CinePulse AI**")
st.sidebar.caption("Movie Sentiment Analysis System")
st.sidebar.divider()

backend_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=DEFAULT_BACKEND_URL,
    help="Address where the FastAPI server is running."
)


@st.cache_data(ttl=5)
def check_backend_health(url: str):
    try:
        r = requests.get(f"{url}/health", timeout=2.5)
        if r.status_code == 200:
            return True, r.json()
        return False, {"error": f"Status {r.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster_bytes(url: str):
    """
    Downloads image bytes server-side using desktop browser User-Agent
    to bypass CDN hotlinking blocks (such as Amazon/IMDb or Wikimedia restrictions).
    Cached for instant display.
    """
    if not url or url == "N/A" or not str(url).startswith("http"):
        return None
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
        }
        resp = requests.get(url, headers=headers, timeout=3.5)
        if resp.status_code == 200 and len(resp.content) > 500:
            return resp.content
    except Exception:
        pass
    return None


def render_cinema_card(title: str, year: str, rating: str, genre: str):
    """Renders a stylish glowing cinema poster card fallback with title, year, genre, and rating."""
    card_html = f"""
    <div style="background: linear-gradient(145deg, #0d1527 0%, #17213c 50%, #0f172a 100%); 
                border: 2px solid #f5c518; border-radius: 14px; padding: 28px 16px; 
                text-align: center; color: #f8fafc; box-shadow: 0 10px 28px rgba(0,0,0,0.65);">
        <div style="font-size: 3.8rem; margin-bottom: 12px; filter: drop-shadow(0 4px 10px rgba(245,197,24,0.45));">🎬</div>
        <h3 style="margin: 0 0 8px 0; color: #f5c518; font-size: 1.3rem; font-weight: 800; letter-spacing: 0.5px;">{title}</h3>
        <div style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 14px; font-weight: 500;">{year} &bull; {genre}</div>
        <div style="display: inline-block; background: rgba(245, 197, 24, 0.15); 
                    border: 1.5px solid rgba(245, 197, 24, 0.65); color: #f5c518; 
                    padding: 6px 18px; border-radius: 20px; font-weight: 800; font-size: 0.95rem;
                    box-shadow: 0 2px 10px rgba(245, 197, 24, 0.2);">
            ⭐ {rating}/10
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


is_online, health_data = check_backend_health(backend_url)

if is_online:
    st.sidebar.success("🟢 **Backend Connected**")
    model_ok = health_data.get("model_loaded", False)
    if model_ok:
        st.sidebar.caption("✅ ML Model Loaded & Active")
    else:
        st.sidebar.warning("⚠️ Model not loaded. Run `train_model.py`.")
else:
    st.sidebar.error("🔴 **Backend Offline**")
    st.sidebar.caption(
        "Start FastAPI backend in your terminal:\n"
        "```bash\nuvicorn main:app --reload --port 8000\n```"
    )

st.sidebar.divider()

# Configurable Decision Threshold
st.sidebar.markdown("### ⚙️ **Threshold Settings**")
threshold_percent = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=50,
    max_value=90,
    value=60,
    step=5,
    help=(
        "If prediction confidence is below this threshold, the review is classified as "
        "Neutral / Uncertain (threshold-based decision) since the model was trained on binary data."
    )
)
confidence_threshold = threshold_percent / 100.0

st.sidebar.caption(
    f"Reviews with confidence < **{threshold_percent}%** are categorized as "
    "**Neutral / Uncertain** to avoid false certainty."
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Project Info:**
    - **Architecture**: Decoupled FastAPI + Streamlit
    - **ML Model**: TF-IDF + Logistic Regression
    - **Dataset**: IMDb Movie Reviews (Stratified 80/20)
    - **Integrations**: OMDb API
    """
)


# ==========================================
# Navigation Tabs
# ==========================================

tabs = st.tabs([
    "🎯 Review Predictor",
    "📂 Batch CSV Analysis",
    "🍿 Movie Explorer (OMDb)",
    "📊 Model Performance"
])


# ==========================================
# TAB 1: Review Predictor
# ==========================================
with tabs[0]:
    st.markdown("### 🎬 Analyze Movie Review Sentiment")
    st.markdown(
        "Enter any movie review below or click one of the preset sample reviews to test real-time sentiment prediction, "
        "calibrated confidence scores, and word attribution explainability."
    )

    # Sample Buttons
    st.markdown("##### ⚡ Quick Test Presets:")
    sample_cols = st.columns(len(SAMPLE_REVIEWS))
    for i, s in enumerate(SAMPLE_REVIEWS):
        if sample_cols[i].button(s["label"], key=f"sample_btn_{i}", use_container_width=True):
            st.session_state.review_input = s["text"]

    # Review text input
    review_text = st.text_area(
        "Movie Review:",
        value=st.session_state.review_input,
        height=140,
        placeholder="Type or paste your movie review here (e.g., 'An unforgettable cinematic masterpiece with brilliant directing...')"
    )

    col_btn, col_stats = st.columns([1, 2])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
    with col_stats:
        words = len(review_text.strip().split()) if review_text.strip() else 0
        chars = len(review_text.strip())
        st.caption(f"📝 **{words}** words | **{chars}** characters")

    if analyze_clicked:
        if not review_text.strip():
            st.warning("⚠️ Please enter a movie review to analyze.")
        elif not is_online:
            st.error("❌ Cannot analyze review: FastAPI backend is offline. Please launch the backend with `uvicorn main:app --port 8000`.")
        else:
            with st.spinner("Analyzing text with TF-IDF and Logistic Regression..."):
                try:
                    payload = {
                        "review": review_text.strip(),
                        "confidence_threshold": confidence_threshold
                    }
                    resp = requests.post(f"{backend_url}/predict", json=payload, timeout=10.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        sentiment = data["sentiment"]
                        conf = data["confidence"] * 100
                        pos_prob = data["positive_probability"] * 100
                        neg_prob = data["negative_probability"] * 100
                        is_unc = data["is_uncertain"]
                        explanation = data["explanation"]
                        keywords = data.get("top_keywords", [])

                        # Save to session history
                        st.session_state.history.insert(0, {
                            "timestamp": time.strftime("%H:%M:%S"),
                            "review": review_text[:120] + ("..." if len(review_text) > 120 else ""),
                            "sentiment": sentiment,
                            "confidence": f"{conf:.1f}%",
                            "pos_score": f"{pos_prob:.1f}%",
                            "neg_score": f"{neg_prob:.1f}%"
                        })

                        st.divider()

                        # Result Banner
                        col_res_badge, col_res_conf, col_res_words = st.columns([1.5, 1.5, 1])

                        with col_res_badge:
                            st.markdown("**Detected Sentiment:**")
                            if sentiment == "Positive":
                                st.markdown('<span class="badge-positive">🍿 POSITIVE</span>', unsafe_allow_html=True)
                            elif sentiment == "Negative":
                                st.markdown('<span class="badge-negative">👎 NEGATIVE</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-neutral">🤔 NEUTRAL / UNCERTAIN</span>', unsafe_allow_html=True)

                        with col_res_conf:
                            st.markdown("**Model Confidence:**")
                            st.markdown(f"### **{conf:.1f}%**")

                        with col_res_words:
                            st.markdown("**Vocabulary Matched:**")
                            st.markdown(f"### **{len(keywords)}** terms")

                        # Dual Gauge Bar
                        st.markdown("**Class Probability Distribution:**")
                        st.markdown(
                            f"""
                            <div class="gauge-wrapper">
                                <div class="gauge-pos" style="width: {pos_prob:.1f}%;" title="Positive: {pos_prob:.1f}%"></div>
                                <div class="gauge-neg" style="width: {neg_prob:.1f}%;" title="Negative: {neg_prob:.1f}%"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8;">
                                <span>🟢 Positive: <b>{pos_prob:.1f}%</b></span>
                                <span>🔴 Negative: <b>{neg_prob:.1f}%</b></span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Decision Rationale
                        st.markdown("---")
                        st.info(f"💡 **Classification Rationale:** {explanation}")

                        # Explainability Word Attribution
                        if keywords:
                            st.markdown("#### 🔍 Explainable AI: Word Influence Attribution")
                            st.caption("Words that exerted the strongest pull on the classifier (Green = Positive influence, Red = Negative influence):")

                            pos_terms = [k for k in keywords if k["sentiment"] == "positive"]
                            neg_terms = [k for k in keywords if k["sentiment"] == "negative"]

                            col_kw_pos, col_kw_neg = st.columns(2)
                            with col_kw_pos:
                                st.markdown("**🟢 Positive Pull Terms:**")
                                if pos_terms:
                                    tags_html = "".join([f'<span class="tag-pos">{t["term"]} (+{t["weight"]:.2f})</span>' for t in pos_terms])
                                    st.markdown(tags_html, unsafe_allow_html=True)
                                else:
                                    st.caption("No strong positive keywords detected.")

                            with col_kw_neg:
                                st.markdown("**🔴 Negative Pull Terms:**")
                                if neg_terms:
                                    tags_html = "".join([f'<span class="tag-neg">{t["term"]} (-{t["weight"]:.2f})</span>' for t in neg_terms])
                                    st.markdown(tags_html, unsafe_allow_html=True)
                                else:
                                    st.caption("No strong negative keywords detected.")
                    else:
                        err_detail = resp.json().get("detail", "Error processing review.")
                        st.error(f"❌ Backend Error ({resp.status_code}): {err_detail}")
                except Exception as ex:
                    st.error(f"❌ Failed to communicate with prediction service: {ex}")

    # Session Prediction History
    if st.session_state.history:
        st.divider()
        st.markdown("#### 📜 Recent Session Prediction History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History", key="clear_hist_btn"):
            st.session_state.history = []
            st.rerun()


# ==========================================
# TAB 2: Batch CSV Analysis
# ==========================================
with tabs[1]:
    st.markdown("### 📂 Batch Movie Review Classification")
    st.markdown(
        "Upload a CSV dataset containing movie reviews to perform bulk sentiment classification, "
        "generate visual distribution metrics, and download the annotated dataset."
    )

    # Sample CSV Download helper
    sample_csv_data = (
        "review\n"
        "\"An absolute masterpiece with breathtaking visuals and stellar acting.\"\n"
        "\"A complete waste of time. The script makes no sense and acting was terrible.\"\n"
        "\"Visually stunning sci-fi spectacle with great philosophical depth.\"\n"
        "\"Dull, lifeless, and painfully boring from start to finish.\"\n"
        "\"The cinematography was decent though the pacing felt slightly slow.\"\n"
    )
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv_data,
        file_name="sample_movie_reviews.csv",
        mime="text/csv",
        help="Download a 5-row sample CSV to test batch classification."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file with movie reviews:",
        type=["csv"],
        help="File must contain a column named 'review', 'text', 'content', or similar."
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with **{len(batch_df)}** rows and columns: `{list(batch_df.columns)}`")

            # Detect review column
            possible_cols = [c for c in batch_df.columns if c.lower() in ["review", "text", "review_text", "reviews", "comment", "content"]]
            default_col = possible_cols[0] if possible_cols else batch_df.columns[0]

            selected_col = st.selectbox(
                "Select the review text column:",
                options=list(batch_df.columns),
                index=list(batch_df.columns).index(default_col)
            )

            st.markdown("##### Preview of uploaded data:")
            st.dataframe(batch_df[[selected_col]].head(5), use_container_width=True)

            if st.button("⚡ Run Batch Classification", type="primary", use_container_width=True):
                if not is_online:
                    st.error("❌ FastAPI backend is offline. Please launch backend to process batch.")
                else:
                    reviews_list = batch_df[selected_col].dropna().astype(str).tolist()
                    if not reviews_list:
                        st.warning("No valid text rows found in selected column.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("Sending reviews to FastAPI backend...")

                        # Send in batches of 50
                        batch_size = 50
                        all_results = []
                        total_items = len(reviews_list)

                        for idx in range(0, total_items, batch_size):
                            chunk = reviews_list[idx:idx + batch_size]
                            req_payload = {
                                "reviews": chunk,
                                "confidence_threshold": confidence_threshold
                            }
                            resp = requests.post(f"{backend_url}/predict/batch", json=req_payload, timeout=30.0)
                            if resp.status_code == 200:
                                res_chunk = resp.json().get("results", [])
                                all_results.extend(res_chunk)
                            progress_bar.progress(min(1.0, (idx + len(chunk)) / total_items))
                            status_text.text(f"Processed {min(total_items, idx + len(chunk))} / {total_items} reviews...")

                        status_text.text("Batch processing complete!")
                        progress_bar.empty()

                        if all_results:
                            res_df = pd.DataFrame(all_results)
                            # Combine with original
                            batch_df["predicted_sentiment"] = [r["sentiment"] for r in all_results]
                            batch_df["confidence_score"] = [r["confidence"] for r in all_results]
                            batch_df["positive_probability"] = [r["positive_probability"] for r in all_results]
                            batch_df["negative_probability"] = [r["negative_probability"] for r in all_results]

                            # Display summary metrics
                            counts = batch_df["predicted_sentiment"].value_counts()
                            pos_c = counts.get("Positive", 0)
                            neg_c = counts.get("Negative", 0)
                            unc_c = counts.get("Neutral / Uncertain", 0)

                            st.markdown("#### 📊 Batch Analysis Summary")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Total Analyzed", len(batch_df))
                            c2.metric("Positive Reviews", f"{pos_c} ({(pos_c/len(batch_df)*100):.1f}%)")
                            c3.metric("Negative Reviews", f"{neg_c} ({(neg_c/len(batch_df)*100):.1f}%)")
                            c4.metric("Neutral / Uncertain", f"{unc_c} ({(unc_c/len(batch_df)*100):.1f}%)")

                            # Bar chart
                            st.bar_chart(counts)

                            # Show annotated table
                            st.markdown("##### Annotated Dataset:")
                            st.dataframe(batch_df.head(20), use_container_width=True)

                            # Download CSV
                            csv_buffer = io.StringIO()
                            batch_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="💾 Download Results as CSV",
                                data=csv_buffer.getvalue(),
                                file_name=f"sentiment_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                type="primary"
                            )
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")


# ==========================================
# TAB 3: Movie Explorer (OMDb Integration)
# ==========================================
with tabs[2]:
    st.markdown("### 🍿 Movie Explorer & Narrative Tone Analysis")
    st.markdown(
        "Search for any movie title to fetch verified metadata from the **OMDb API** "
        "and automatically analyze the sentiment tone of the movie's storyline using our ML model."
    )

    # Load all Hindi movies for dropdown
    all_hindi_titles = []
    hindi_json_file = os.path.join(os.path.dirname(__file__), "data", "hindi_movies.json")
    if os.path.exists(hindi_json_file):
        try:
            import json
            with open(hindi_json_file, "r", encoding="utf-8") as f:
                h_data = json.load(f)
                all_hindi_titles = sorted(list({m["Title"] for m in h_data.values()}))
        except Exception:
            all_hindi_titles = []

    # Quick movie test presets - Bollywood & International
    if "movie_title_input" not in st.session_state:
        st.session_state.movie_title_input = ""
    if "execute_search" not in st.session_state:
        st.session_state.execute_search = False
    if "searched_movie_data" not in st.session_state:
        st.session_state.searched_movie_data = None

    st.markdown("##### 🇮🇳 Top Bollywood Hits:")
    hindi_cols = st.columns(6)
    hindi_movies = [
        "12th Fail",
        "Stree 2",
        "3 Idiots",
        "Dangal",
        "Tumbbad",
        "Jawan"
    ]
    for idx, hm in enumerate(hindi_movies):
        if hindi_cols[idx].button(f"🇮🇳 {hm}", key=f"hindi_preset_{idx}", use_container_width=True):
            st.session_state.movie_title_input = hm
            st.session_state.execute_search = True

    if all_hindi_titles:
        selected_from_catalog = st.selectbox(
            "📋 Or Choose directly from All Hindi Movies Catalog (40+ Films):",
            options=["-- Click here to select any Hindi movie --"] + all_hindi_titles,
            key="hindi_dropdown"
        )
        if selected_from_catalog != "-- Click here to select any Hindi movie --":
            if st.session_state.get("prev_dropdown_selection") != selected_from_catalog:
                st.session_state.prev_dropdown_selection = selected_from_catalog
                st.session_state.movie_title_input = selected_from_catalog
                st.session_state.execute_search = True

    st.markdown("##### 🌍 International Cinema Presets:")
    hollywood_cols = st.columns(6)
    hollywood_movies = [
        "Inception",
        "The Dark Knight",
        "Interstellar",
        "Titanic",
        "The Godfather",
        "Oppenheimer"
    ]
    for idx, pm in enumerate(hollywood_movies):
        if hollywood_cols[idx].button(f"🎬 {pm}", key=f"hollywood_preset_{idx}", use_container_width=True):
            st.session_state.movie_title_input = pm
            st.session_state.execute_search = True

    movie_col_input, movie_col_btn = st.columns([3, 1])
    with movie_col_input:
        movie_title_query = st.text_input(
            "Search Movie Title:",
            value=st.session_state.movie_title_input,
            placeholder="e.g. Inception, The Dark Knight, The Godfather, Titanic, Parasite"
        )
    with movie_col_btn:
        st.write("")
        st.write("")
        search_movie_btn = st.button("🔍 Search Movie", type="primary", use_container_width=True)

    do_search = search_movie_btn or st.session_state.execute_search
    st.session_state.execute_search = False

    if do_search:
        search_target = movie_title_query.strip() or st.session_state.movie_title_input.strip()
        if not search_target:
            st.warning("Please enter a movie title to search.")
        elif not is_online:
            st.error("FastAPI backend is offline. Start the backend to search movies.")
        else:
            with st.spinner(f"Retrieving '{search_target}' movie data..."):
                try:
                    resp = requests.get(f"{backend_url}/movie/{search_target}", timeout=10.0)
                    if resp.status_code == 200:
                        st.session_state.searched_movie_data = resp.json()
                    else:
                        st.error(f"Backend returned status {resp.status_code}.")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    if st.session_state.searched_movie_data:
        data = st.session_state.searched_movie_data
        status_str = data.get("status")

        if status_str == "unconfigured":
            st.warning(
                "⚙️ **OMDb API Key Missing**\n\n"
                f"{data.get('message')}\n\n"
                "**How to configure:**\n"
                "1. Get a free key at [omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx)\n"
                "2. Add `OMDB_API_KEY=your_key_here` to your `.env` file\n"
                "3. Restart the FastAPI backend."
            )
        elif status_str == "not_found":
            st.error(f"❌ {data.get('message')}")
        elif status_str == "success":
            movie = data.get("movie", {})
            plot_analysis = data.get("plot_sentiment")

            st.success(f"Found: **{movie.get('Title')} ({movie.get('Year')})**")

            col_poster, col_details = st.columns([1, 2])

            with col_poster:
                poster_url = movie.get("Poster")
                mov_title = movie.get("Title", "Movie")
                mov_year = movie.get("Year", "")
                mov_rating = movie.get("imdbRating", "N/A")
                mov_genre = movie.get("Genre", "Cinema")

                # Try server-side cached download first to bypass browser CDN hotlink blocks
                poster_bytes = fetch_poster_bytes(poster_url)
                if poster_bytes:
                    try:
                        st.image(poster_bytes, caption=f"{mov_title} ({mov_year})", use_container_width=True)
                    except Exception:
                        render_cinema_card(mov_title, mov_year, mov_rating, mov_genre)
                elif poster_url and poster_url != "N/A" and str(poster_url).startswith("http"):
                    try:
                        st.image(poster_url, caption=f"{mov_title} ({mov_year})", use_container_width=True)
                    except Exception:
                        render_cinema_card(mov_title, mov_year, mov_rating, mov_genre)
                else:
                    render_cinema_card(mov_title, mov_year, mov_rating, mov_genre)

            with col_details:
                st.markdown(f"## {movie.get('Title')} ({movie.get('Year')})")
                st.markdown(f"**Genre:** `{movie.get('Genre')}` | **Runtime:** `{movie.get('Runtime')}` | **Rated:** `{movie.get('Rated')}`")
                st.markdown(f"⭐ **IMDb Rating:** `{movie.get('imdbRating')}/10` ({movie.get('imdbVotes')} votes) | 🏆 **Metascore:** `{movie.get('Metascore')}`")
                st.markdown(f"🎬 **Director:** {movie.get('Director')}")
                st.markdown(f"✍️ **Writers:** {movie.get('Writer')}")
                st.markdown(f"🌟 **Cast:** {movie.get('Actors')}")
                st.markdown(f"💰 **Box Office:** {movie.get('BoxOffice', 'N/A')}")
                st.markdown(f"🎖️ **Awards:** {movie.get('Awards', 'N/A')}")

            st.divider()
            st.markdown("#### 📖 Full Plot Summary")
            st.write(movie.get("Plot", "No plot provided."))

            # Narrative Tone Sentiment from ML Model
            if plot_analysis:
                st.markdown("#### 🎭 Narrative Plot Tone (Machine Learning Analysis)")
                p_sent = plot_analysis.get("sentiment")
                p_conf = plot_analysis.get("confidence", 0) * 100
                p_pos = plot_analysis.get("positive_probability", 0) * 100
                p_neg = plot_analysis.get("negative_probability", 0) * 100

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    if p_sent == "Positive":
                        st.markdown(f"**Plot Tone:** <span class='badge-positive'>UPLIFTING / HEROIC ({p_conf:.1f}% confidence)</span>", unsafe_allow_html=True)
                    elif p_sent == "Negative":
                        st.markdown(f"**Plot Tone:** <span class='badge-negative'>DARK / DRAMATIC / CONFLICT-HEAVY ({p_conf:.1f}% confidence)</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Plot Tone:** <span class='badge-neutral'>BALANCED / NEUTRAL TONE</span>", unsafe_allow_html=True)

                with col_p2:
                    st.caption(f"Positive signals: {p_pos:.1f}% | Dark/conflict signals: {p_neg:.1f}%")

                keywords = plot_analysis.get("top_keywords", [])
                if keywords:
                    st.caption("Key thematic vocabulary extracted from plot:")
                    tags = " ".join([f"`{k['term']}`" for k in keywords[:6]])
                    st.markdown(tags)


# ==========================================
# TAB 4: Model Performance & Metrics
# ==========================================
with tabs[3]:
    st.markdown("### 📊 Machine Learning Model Evaluation")
    st.markdown(
        "Actual, verified metrics calculated on the 20% holdout test set using an **80/20 Stratified Split**. "
        "No fabricated values — all metrics reflect genuine scikit-learn model evaluation."
    )

    metrics_loaded = False
    metrics = None

    if is_online:
        try:
            m_resp = requests.get(f"{backend_url}/metrics", timeout=5.0)
            if m_resp.status_code == 200:
                metrics = m_resp.json().get("metrics")
                metrics_loaded = True
        except Exception:
            metrics_loaded = False

    # Fallback to local file if backend offline
    if not metrics_loaded:
        from ml_service import load_metrics, DEFAULT_METRICS_PATH
        if os.path.exists(DEFAULT_METRICS_PATH):
            try:
                metrics = load_metrics()
                metrics_loaded = True
            except Exception:
                metrics_loaded = False

    if not metrics_loaded or not metrics:
        st.warning(
            "⚠️ **Evaluation metrics not found.**\n\n"
            "Please train the model first by running:\n"
            "```bash\npython train_model.py\n```"
        )
    else:
        # Display KPI metric cards
        acc = metrics.get("accuracy", 0.0) * 100
        prec = metrics.get("precision", 0.0) * 100
        rec = metrics.get("recall", 0.0) * 100
        f1 = metrics.get("f1_score", 0.0) * 100
        f1_macro = metrics.get("f1_macro", 0.0) * 100
        vocab_size = metrics.get("vocabulary_size", 0)

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("🎯 Accuracy", f"{acc:.2f}%")
        col_m2.metric("📐 Precision", f"{prec:.2f}%")
        col_m3.metric("🎯 Recall", f"{rec:.2f}%")
        col_m4.metric("⚖️ F1-Score (Pos)", f"{f1:.2f}%")
        col_m5.metric("🌐 Macro F1", f"{f1_macro:.2f}%")

        st.divider()

        # Confusion Matrix & Classification Details
        col_cm, col_specs = st.columns([1.2, 1])

        with col_cm:
            st.markdown("#### 🧩 Confusion Matrix (Holdout Test Set)")
            cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))

            # Plot heatmap using matplotlib & seaborn
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#131b2e")
            ax.set_facecolor("#131b2e")

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Pred Negative", "Pred Positive"],
                yticklabels=["Actual Negative", "Actual Positive"],
                cbar=False,
                ax=ax,
                annot_kws={"size": 14, "weight": "bold", "color": "#f8fafc"}
            )
            ax.tick_params(colors="#94a3b8", labelsize=10)
            ax.set_title("Confusion Matrix", color="#f8fafc", pad=12, fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)

        with col_specs:
            st.markdown("#### ⚙️ Pipeline Specifications")
            st.markdown(
                f"""
                - **Algorithm**: `LogisticRegression(C=2.0, class_weight='balanced')`
                - **Feature Extractor**: `TfidfVectorizer(ngram_range=(1, 2))`
                - **Vocabulary Size**: `{vocab_size:,}` n-grams
                - **Dataset Name**: `{metrics.get('dataset_name', 'imdb_dataset.csv')}`
                - **Total Samples**: `{metrics.get('total_samples', 0):,}` reviews
                - **Train / Test Split**: `{metrics.get('split_ratio', '80/20')}` (Stratified)
                - **Training Samples**: `{metrics.get('train_samples', 0):,}`
                - **Testing Samples**: `{metrics.get('test_samples', 0):,}`
                - **Classes**: `negative`, `positive`
                """
            )

        # Classification Report Table
        st.divider()
        st.markdown("#### 📋 Detailed Classification Report")
        report = metrics.get("classification_report", {})
        if report:
            table_rows = []
            for class_key in ["negative", "positive", "macro avg", "weighted avg"]:
                if class_key in report:
                    row_data = report[class_key]
                    table_rows.append({
                        "Class / Metric": class_key.title(),
                        "Precision": f"{row_data.get('precision', 0)*100:.2f}%",
                        "Recall": f"{row_data.get('recall', 0)*100:.2f}%",
                        "F1-Score": f"{row_data.get('f1-score', 0)*100:.2f}%",
                        "Support (Samples)": int(row_data.get("support", 0))
                    })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "🎬 <b>CinePulse Movie Sentiment AI</b> — Minor Project Demonstration &bull; "
    "FastAPI + Streamlit + Scikit-Learn + OMDb"
    "</div>",
    unsafe_allow_html=True
)
