import pandas as pd
from utils import normalize_text, ensure_year_column


class MovieSearchEngine:
    """Search and filter movie metadata."""

    def __init__(self, movies_df):
        self.movies_df = ensure_year_column(movies_df)

    def all_genres(self):
        genres = set()
        for value in self.movies_df["genres"].dropna():
            for genre in str(value).split("|"):
                if genre and genre != "(no genres listed)":
                    genres.add(genre)
        return sorted(genres)

    def search(
        self,
        title_query="",
        genre="All",
        year=None,
        min_year=None,
        max_year=None,
        limit=50,
    ):
        results = self.movies_df.copy()

        if title_query:
            results = results[
                results["title"].str.contains(title_query, case=False, na=False)
            ]

        if genre and genre != "All":
            results = results[
                results["genres"].str.contains(genre, case=False, na=False)
            ]

        if year not in [None, "", "All"]:
            try:
                year = int(year)
                results = results[results["year"] == year]
            except ValueError:
                pass

        if min_year not in [None, ""]:
            try:
                results = results[results["year"] >= int(min_year)]
            except ValueError:
                pass

        if max_year not in [None, ""]:
            try:
                results = results[results["year"] <= int(max_year)]
            except ValueError:
                pass

        return results.head(limit).reset_index(drop=True)

    def title_matches(self, query, limit=10):
        if not query:
            return self.movies_df.head(limit).reset_index(drop=True)

        results = self.movies_df[
            self.movies_df["title"].str.contains(query, case=False, na=False)
        ]

        return results.head(limit).reset_index(drop=True)

    def movies_by_genre(self, genre, limit=25):
        if genre == "All":
            return self.movies_df.head(limit).reset_index(drop=True)

        return self.movies_df[
            self.movies_df["genres"].str.contains(genre, case=False, na=False)
        ].head(limit).reset_index(drop=True)
