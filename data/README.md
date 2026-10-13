# Kaggle IMDB Dataset Instructions

This application supports the popular Kaggle dataset:
**IMDB Dataset of 50K Movie Reviews** (by Lakshmipathi N).

### Dataset Format
The dataset should be a CSV containing two columns:
- `review`: The movie review text.
- `sentiment`: The binary classification label (`positive` or `negative`).

### Using Your Own Kaggle CSV:
1. Download `IMDB Dataset.csv` from Kaggle:
   https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Place the file inside this directory:
   `movie-sentiment-app/data/IMDB Dataset.csv` (or `imdb_dataset.csv`).
3. Run the training script from the project root:
   ```bash
   python train.py
   ```
4. The model will automatically train on the full 50,000 reviews and save the updated pipeline in `models/sentiment_model.joblib`.
