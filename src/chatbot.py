import re
import pandas as pd
from utils import extract_year


class MovieChatbot:
    """
    Lightweight rules-based movie chatbot.

    This is intentionally simple, transparent, and portfolio-friendly.
    It does not require an API key.
    """

    GENRES = [
        "action",
        "adventure",
        "animation",
        "children",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film-noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci-fi",
        "thriller",
        "war",
        "western",
    ]

    MOOD_TO_GENRES = {
        "funny": ["Comedy"],
        "scary": ["Horror", "Thriller"],
        "romantic": ["Romance"],
        "sad": ["Drama"],
        "exciting": ["Action", "Adventure"],
        "family": ["Children", "Animation"],
        "date": ["Romance", "Comedy"],
        "mind": ["Mystery", "Sci-Fi", "Thriller"],
    }

    def __init__(self, model, movies_df):
        self.model = model
        self.movies_df = movies_df.copy()

    def parse_query(self, query):
        query_lower = query.lower()

        selected_genres = []

        for genre in self.GENRES:
            if genre in query_lower:
                selected_genres.append(self._title_case_genre(genre))

        for keyword, genres in self.MOOD_TO_GENRES.items():
            if keyword in query_lower:
                selected_genres.extend(genres)

        selected_genres = sorted(set(selected_genres))

        year_match = re.search(r"(19\d{2}|20\d{2})", query_lower)
        year = int(year_match.group(1)) if year_match else None

        decade = None
        decade_match = re.search(r"(\d{2})s", query_lower)
        if decade_match:
            val = int(decade_match.group(1))
            if val >= 30:
                decade = 1900 + val
            else:
                decade = 2000 + val

        return {
            "genres": selected_genres,
            "year": year,
            "decade": decade,
            "wants_popular": any(word in query_lower for word in ["top", "best", "popular", "highest"]),
        }

    def _title_case_genre(self, genre):
        mapping = {
            "sci-fi": "Sci-Fi",
            "film-noir": "Film-Noir",
        }
        return mapping.get(genre, genre.title())

    def recommend(self, query, top_n=10):
        parsed = self.parse_query(query)
        recs = self.model.popularity_df.copy()

        if "year" not in recs.columns:
            recs["year"] = recs["title"].apply(extract_year)

        for genre in parsed["genres"]:
            recs = recs[recs["genres"].str.contains(genre, case=False, na=False)]

        if parsed["year"]:
            recs = recs[recs["year"] == parsed["year"]]

        if parsed["decade"]:
            recs = recs[
                (recs["year"] >= parsed["decade"])
                & (recs["year"] <= parsed["decade"] + 9)
            ]

        recs = recs.sort_values("weighted_score", ascending=False)

        response = self.explain_response(parsed, len(recs))

        return response, recs.head(top_n).reset_index(drop=True)

    def explain_response(self, parsed, count):
        pieces = []

        if parsed["genres"]:
            pieces.append("genres: " + ", ".join(parsed["genres"]))

        if parsed["year"]:
            pieces.append(f"year: {parsed['year']}")

        if parsed["decade"]:
            pieces.append(f"decade: {parsed['decade']}s")

        if pieces:
            return f"I found {count} movies matching " + " | ".join(pieces) + "."

        return f"I found {count} popular movies based on your request."
