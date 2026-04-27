import re
import pandas as pd


def extract_year(title):
    """Extract release year from a MovieLens title like 'Toy Story (1995)'."""
    match = re.search(r"\((\d{4})\)", str(title))
    return int(match.group(1)) if match else None


def normalize_text(text):
    """Lowercase and trim text safely."""
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def ensure_year_column(movies_df):
    """Add a year column if missing."""
    movies_df = movies_df.copy()
    if "year" not in movies_df.columns:
        movies_df["year"] = movies_df["title"].apply(extract_year)
    return movies_df


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default
