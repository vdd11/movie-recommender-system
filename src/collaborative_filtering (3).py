
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class HybridRecommender:
    """
    Hybrid movie recommender:
    1) Item-item collaborative filtering using cosine KNN
    2) Genre-based user taste profile for re-ranking
    3) Popularity-based fallback for cold-start and tie-breaking
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        genre_df: pd.DataFrame,
        neighbor_k: int = 15,
        knn_neighbors: int = 30,
        min_similarity: float = 0.10,
        min_rating_for_profile: float = 4.0,
        genre_weight: float = 0.35,
        popularity_weight: float = 0.15,
        min_candidate_rating_count: int = 3,
        confidence_scaling: bool = True,
    ) -> None:
        self.train = train_df.copy()
        self.movies = movies_df.copy()
        self.genre_matrix = genre_df.copy()

        self.neighbor_k = neighbor_k
        self.knn_neighbors = knn_neighbors
        self.min_similarity = min_similarity
        self.min_rating_for_profile = min_rating_for_profile
        self.genre_weight = genre_weight
        self.popularity_weight = popularity_weight
        self.min_candidate_rating_count = min_candidate_rating_count
        self.confidence_scaling = confidence_scaling

        self.genre_cols = [c for c in self.genre_matrix.columns if c != "movieId"]

        self.movie_meta = (
            self.movies[["movieId", "title", "genres"]]
            .drop_duplicates()
            .set_index("movieId")
        )

        self.genre_lookup = (
            self.genre_matrix.set_index("movieId")[self.genre_cols]
            .copy()
        )

        self.user_item_train: Optional[pd.DataFrame] = None
        self.user_item_train_filled: Optional[pd.DataFrame] = None
        self.item_user_matrix: Optional[pd.DataFrame] = None
        self.item_user_sparse: Optional[csr_matrix] = None

        self.user_ids = None
        self.movie_ids = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.idx_to_user = None
        self.idx_to_movie = None

        self.knn_model: Optional[NearestNeighbors] = None
        self.popularity_df: Optional[pd.DataFrame] = None
        self.popularity_lookup: Optional[pd.DataFrame] = None
        self.global_mean: Optional[float] = None
        self.existing_user_ids = set()

    def fit(self):
        self._build_user_item_matrix()
        self._build_knn_model()
        self._build_popularity_model()
        self.existing_user_ids = set(self.train["userId"].unique())
        return self

    def _build_user_item_matrix(self) -> None:
        self.user_item_train = self.train.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        )
        self.user_item_train_filled = self.user_item_train.fillna(0)

        self.user_ids = self.user_item_train.index.to_list()
        self.movie_ids = self.user_item_train.columns.to_list()

        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.idx_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_idx.items()}

    def _build_knn_model(self) -> None:
        self.item_user_matrix = self.user_item_train_filled.T
        self.item_user_sparse = csr_matrix(self.item_user_matrix.to_numpy())

        self.knn_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=self.knn_neighbors,
            n_jobs=-1
        )
        self.knn_model.fit(self.item_user_sparse)

    def _build_popularity_model(self) -> None:
        popularity_df = (
            self.train.groupby("movieId")
            .agg(
                avg_rating=("rating", "mean"),
                rating_count=("rating", "count")
            )
            .reset_index()
        )

        self.global_mean = float(self.train["rating"].mean())
        vote_weight = float(popularity_df["rating_count"].mean())

        popularity_df["weighted_score"] = (
            (popularity_df["rating_count"] / (popularity_df["rating_count"] + vote_weight)) * popularity_df["avg_rating"]
            + (vote_weight / (popularity_df["rating_count"] + vote_weight)) * self.global_mean
        )

        popularity_df = popularity_df.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        ).sort_values("weighted_score", ascending=False).reset_index(drop=True)

        self.popularity_df = popularity_df
        self.popularity_lookup = popularity_df.set_index("movieId")

    def get_similar_movies(self, movie_id: int, n_neighbors: Optional[int] = None) -> pd.DataFrame:
        if n_neighbors is None:
            n_neighbors = self.neighbor_k

        if movie_id not in self.movie_to_idx:
            return pd.DataFrame(columns=["movieId", "title", "genres", "similarity"])

        movie_idx = self.movie_to_idx[movie_id]

        distances, indices = self.knn_model.kneighbors(
            self.item_user_sparse[movie_idx],
            n_neighbors=min(n_neighbors + 1, self.item_user_sparse.shape[0])
        )

        rows = []
        for dist, idx in zip(distances.flatten(), indices.flatten()):
            neighbor_movie_id = self.idx_to_movie[idx]
            if neighbor_movie_id == movie_id:
                continue

            similarity = 1 - dist
            rows.append((neighbor_movie_id, similarity))

        neighbors_df = pd.DataFrame(rows, columns=["movieId", "similarity"])

        neighbors_df = neighbors_df.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        ).sort_values("similarity", ascending=False).reset_index(drop=True)

        return neighbors_df

    def build_user_genre_profile(self, user_id: int) -> pd.Series:
        user_likes = self.train[
            (self.train["userId"] == user_id) &
            (self.train["rating"] >= self.min_rating_for_profile)
        ][["movieId", "rating"]].copy()

        if user_likes.empty:
            return pd.Series(0.0, index=self.genre_cols)

        liked_with_genres = user_likes.merge(
            self.genre_matrix[["movieId"] + self.genre_cols],
            on="movieId",
            how="left"
        ).fillna(0)

        weighted_genres = liked_with_genres[self.genre_cols].multiply(
            liked_with_genres["rating"], axis=0
        ).sum()

        if weighted_genres.max() > 0:
            weighted_genres = weighted_genres / weighted_genres.max()

        return weighted_genres

    def get_genre_overlap_score(self, movie_id: int, user_profile: pd.Series) -> float:
        if movie_id not in self.genre_lookup.index:
            return 0.0

        movie_vec = self.genre_lookup.loc[movie_id].astype(float)

        if movie_vec.sum() == 0:
            return 0.0

        overlap = np.dot(movie_vec.values, user_profile.values) / movie_vec.sum()
        return float(overlap)

    def _popularity_fallback(self, top_n: int = 10, exclude_movie_ids=None) -> pd.DataFrame:
        fallback = self.popularity_df.copy()
        if exclude_movie_ids:
            fallback = fallback[~fallback["movieId"].isin(exclude_movie_ids)]

        return fallback.head(top_n)[
            ["movieId", "title", "genres", "weighted_score", "rating_count"]
        ].reset_index(drop=True)

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        if user_id not in self.existing_user_ids:
            return self._popularity_fallback(top_n=top_n)

        user_history = self.train[self.train["userId"] == user_id].copy()
        if user_history.empty:
            return self._popularity_fallback(top_n=top_n)

        rated_movie_ids = set(user_history["movieId"].tolist())
        user_profile = self.build_user_genre_profile(user_id)

        candidate_scores = {}

        for _, row in user_history.iterrows():
            seed_movie_id = row["movieId"]
            user_rating = row["rating"]

            similar_movies = self.get_similar_movies(
                movie_id=seed_movie_id,
                n_neighbors=self.neighbor_k
            )

            for _, sim_row in similar_movies.iterrows():
                candidate_movie_id = sim_row["movieId"]
                similarity = sim_row["similarity"]

                if candidate_movie_id in rated_movie_ids:
                    continue

                if similarity < self.min_similarity:
                    continue

                weighted_contribution = similarity * user_rating

                if candidate_movie_id not in candidate_scores:
                    candidate_scores[candidate_movie_id] = {
                        "weighted_sum": 0.0,
                        "similarity_sum": 0.0,
                        "reasons": []
                    }

                candidate_scores[candidate_movie_id]["weighted_sum"] += weighted_contribution
                candidate_scores[candidate_movie_id]["similarity_sum"] += similarity
                candidate_scores[candidate_movie_id]["reasons"].append(
                    (seed_movie_id, similarity, user_rating)
                )

        if not candidate_scores:
            return self._popularity_fallback(top_n=top_n, exclude_movie_ids=rated_movie_ids)

        rows = []
        for candidate_movie_id, values in candidate_scores.items():
            cf_score = values["weighted_sum"] / values["similarity_sum"]

            genre_score = self.get_genre_overlap_score(
                movie_id=candidate_movie_id,
                user_profile=user_profile
            )

            if candidate_movie_id in self.popularity_lookup.index:
                pop_score = float(self.popularity_lookup.loc[candidate_movie_id, "weighted_score"])
                rating_count = int(self.popularity_lookup.loc[candidate_movie_id, "rating_count"])
            else:
                pop_score = self.global_mean
                rating_count = 0

            if rating_count < self.min_candidate_rating_count:
                continue

            pop_score_norm = pop_score / 5.0

            hybrid_score = (
                cf_score
                + self.genre_weight * genre_score
                + self.popularity_weight * pop_score_norm
            )

            if self.confidence_scaling:
                confidence_boost = min(rating_count / 50.0, 1.0)
                hybrid_score = hybrid_score * (0.85 + 0.15 * confidence_boost)

            best_reason = sorted(
                values["reasons"],
                key=lambda x: x[1] * x[2],
                reverse=True
            )[0]

            reason_movie_id = best_reason[0]
            reason_title = (
                self.movie_meta.loc[reason_movie_id, "title"]
                if reason_movie_id in self.movie_meta.index
                else str(reason_movie_id)
            )

            rows.append({
                "movieId": candidate_movie_id,
                "cf_score": round(cf_score, 4),
                "genre_score": round(genre_score, 4),
                "popularity_score": round(pop_score, 4),
                "hybrid_score": round(hybrid_score, 4),
                "rating_count": int(rating_count),
                "reason": f"Because you liked: {reason_title}"
            })

        if not rows:
            return self._popularity_fallback(top_n=top_n, exclude_movie_ids=rated_movie_ids)

        recs = pd.DataFrame(rows)

        recs = recs.merge(
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        recs = recs.sort_values(
            ["hybrid_score", "cf_score", "popularity_score", "rating_count"],
            ascending=False
        ).head(top_n).reset_index(drop=True)

        return recs[
            [
                "movieId",
                "title",
                "genres",
                "cf_score",
                "genre_score",
                "popularity_score",
                "hybrid_score",
                "rating_count",
                "reason"
            ]
        ]

    def explain_recommendation(self, user_id: int, recommended_movie_id: int) -> pd.DataFrame:
        if user_id not in self.existing_user_ids:
            return pd.DataFrame(columns=["because_you_rated", "your_rating", "similarity"])

        user_history = self.train[self.train["userId"] == user_id].copy()
        explanations = []

        for _, row in user_history.iterrows():
            seed_movie_id = row["movieId"]
            user_rating = row["rating"]

            similar_movies = self.get_similar_movies(seed_movie_id, n_neighbors=self.neighbor_k)

            match = similar_movies[similar_movies["movieId"] == recommended_movie_id]
            if not match.empty:
                sim = match.iloc[0]["similarity"]

                seed_title = (
                    self.movie_meta.loc[seed_movie_id, "title"]
                    if seed_movie_id in self.movie_meta.index
                    else str(seed_movie_id)
                )

                explanations.append({
                    "because_you_rated": seed_title,
                    "your_rating": user_rating,
                    "similarity": round(float(sim), 4)
                })

        if not explanations:
            return pd.DataFrame(columns=["because_you_rated", "your_rating", "similarity"])

        return pd.DataFrame(explanations).sort_values(
            "similarity", ascending=False
        ).reset_index(drop=True)

    def explain_recommendation_text(self, user_id: int, recommended_movie_id: int) -> str:
        explanation_df = self.explain_recommendation(user_id, recommended_movie_id)

        if explanation_df.empty:
            return "No clear explanation found."

        top_reason = explanation_df.iloc[0]
        return (
            f"Recommended because you rated '{top_reason['because_you_rated']}' highly "
            f"(your rating: {top_reason['your_rating']}, similarity: {top_reason['similarity']})."
        )

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)


def load_processed_artifacts(base_dir: str):
    processed_dir = os.path.join(base_dir, "data", "processed")

    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    movies_clean = pd.read_csv(os.path.join(processed_dir, "movies_clean.csv"))
    genre_encoded = pd.read_csv(os.path.join(processed_dir, "genre_encoded.csv"))

    for df, col in [(train, "rated_at"), (val, "rated_at"), (test, "rated_at")]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return train, val, test, movies_clean, genre_encoded
