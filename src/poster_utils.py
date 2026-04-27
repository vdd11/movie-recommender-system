"""
Poster helpers.

This file is intentionally API-safe. It does not require a paid API key.
If you later enrich movies_clean.csv with a poster_url column, the app will
automatically display real posters.
"""


def get_poster_url(movie_row):
    """
    Return poster URL if the row has one.
    Supported optional columns:
    - poster_url
    - poster_path
    """
    if movie_row is None:
        return None

    if "poster_url" in movie_row and isinstance(movie_row["poster_url"], str) and movie_row["poster_url"].startswith("http"):
        return movie_row["poster_url"]

    if "poster_path" in movie_row and isinstance(movie_row["poster_path"], str) and movie_row["poster_path"]:
        path = movie_row["poster_path"]
        if path.startswith("http"):
            return path
        return f"https://image.tmdb.org/t/p/w342{path}"

    return None


def poster_caption(title):
    return f"Poster for {title}" if title else "Movie poster"
