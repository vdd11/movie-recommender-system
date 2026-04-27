import os
import pandas as pd

from config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    MOVIES_PATH,
    GENRES_PATH,
    REQUIRED_PROCESSED_FILES,
)
from utils import ensure_year_column


def validate_processed_files():
    missing = [path for path in REQUIRED_PROCESSED_FILES if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing processed files:\n"
            + "\n".join(missing)
            + "\n\nRun your cleaning notebook first so data/processed contains train.csv, val.csv, test.csv, movies_clean.csv, and genre_encoded.csv."
        )


def load_processed_data():
    validate_processed_files()

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    genres = pd.read_csv(GENRES_PATH)

    for df in [train, val, test]:
        if "rated_at" in df.columns:
            df["rated_at"] = pd.to_datetime(df["rated_at"], errors="coerce")

    movies = ensure_year_column(movies)

    return train, val, test, movies, genres
