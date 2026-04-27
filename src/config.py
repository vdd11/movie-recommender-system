"""
Central project configuration.

Edit BASE_DIR only if your repository path is different.
Most files assume this structure:

movie-recommender-system/
├── app/
├── src/
├── data/
│   ├── processed/
│   └── user/
├── models/
└── requirements.txt
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
USER_DATA_DIR = os.path.join(DATA_DIR, "user")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_recommender.pkl")
USER_RATINGS_PATH = os.path.join(USER_DATA_DIR, "user_ratings.csv")

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
MOVIES_PATH = os.path.join(PROCESSED_DIR, "movies_clean.csv")
GENRES_PATH = os.path.join(PROCESSED_DIR, "genre_encoded.csv")

DEFAULT_TOP_N = 10
DEFAULT_SAMPLE_USERS = 100
DEFAULT_MIN_RELEVANT_RATING = 4.0

RECOMMENDER_PARAMS = {
    "n_components": 50,
    "neighbor_k": 15,
    "min_similarity": 0.10,
    "min_rating_for_profile": 4.0,
    "cf_weight": 0.30,
    "mf_weight": 0.45,
    "genre_weight": 0.15,
    "popularity_weight": 0.10,
}

REQUIRED_PROCESSED_FILES = [
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    MOVIES_PATH,
    GENRES_PATH,
]
