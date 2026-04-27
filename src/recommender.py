import os
import pickle
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


class HybridRecommender:
    def __init__(
        self,
        train_df,
        movies_df,
        genre_df,
        n_components=50,
        neighbor_k=15,
        min_similarity=0.10,
        min_rating_for_profile=4.0,
        cf_weight=0.30,
        mf_weight=0.45,
        genre_weight=0.15,
        popularity_weight=0.10,
    ):
        self.train_df = train_df.copy()
        self.movies_df = movies_df.copy()
        self.genre_df = genre_df.copy()

        self.n_components = n_components
        self.neighbor_k = neighbor_k
        self.min_similarity = min_similarity
        self.min_rating_for_profile = min_rating_for_profile

        self.cf_weight = cf_weight
        self.mf_weight = mf_weight
        self.genre_weight = genre_weight
        self.popularity_weight = popularity_weight

    def fit(self):
        self.user_item = self.train_df.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
        )

        self.user_item_filled = self.user_item.fillna(0)

        self.user_ids = self.user_item.index.tolist()
        self.movie_ids = self.user_item.columns.tolist()

        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}

        self.matrix_sparse = csr_matrix(self.user_item_filled.values)

        n_features = min(self.user_item_filled.shape) - 1
        safe_components = max(1, min(self.n_components, n_features))

        self.svd = TruncatedSVD(
            n_components=safe_components,
            random_state=42,
        )

        self.user_factors = self.svd.fit_transform(self.matrix_sparse)
        self.movie_factors = self.svd.components_.T

        self.pred_matrix = np.dot(self.user_factors, self.movie_factors.T)

        self.pred_df = pd.DataFrame(
            self.pred_matrix,
            index=self.user_ids,
            columns=self.movie_ids,
        )

        self.item_user_matrix = self.user_item_filled.T
        self.item_user_sparse = csr_matrix(self.item_user_matrix.values)

        self.knn_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(50, len(self.movie_ids)),
            n_jobs=-1,
        )
        self.knn_model.fit(self.item_user_sparse)

        self._build_metadata()
        self._build_popularity()
        self._build_genres()

        return self

    def _build_metadata(self):
        self.movie_meta = (
            self.movies_df[["movieId", "title", "genres"]]
            .drop_duplicates()
            .set_index("movieId")
        )

    def _build_popularity(self):
        self.popularity_df = (
            self.train_df.groupby("movieId")
            .agg(
                avg_rating=("rating", "mean"),
                rating_count=("rating", "count"),
            )
            .reset_index()
        )

        self.global_mean = self.train_df["rating"].mean()
        vote_weight = self.popularity_df["rating_count"].mean()

        self.popularity_df["weighted_score"] = (
            (self.popularity_df["rating_count"] / (self.popularity_df["rating_count"] + vote_weight))
            * self.popularity_df["avg_rating"]
            + (vote_weight / (self.popularity_df["rating_count"] + vote_weight))
            * self.global_mean
        )

        self.popularity_df = self.popularity_df.merge(
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        )

        self.popularity_df = self.popularity_df.sort_values(
            "weighted_score",
            ascending=False,
        ).reset_index(drop=True)

        self.popularity_lookup = self.popularity_df.set_index("movieId")

    def _build_genres(self):
        self.genre_cols = [c for c in self.genre_df.columns if c != "movieId"]
        self.genre_lookup = self.genre_df.set_index("movieId")[self.genre_cols].copy()

    def _popularity_fallback(self, top_n=10, exclude_ids=None):
        exclude_ids = exclude_ids or set()
        recs = self.popularity_df[
            ~self.popularity_df["movieId"].isin(exclude_ids)
        ].copy()
        return recs.head(top_n).reset_index(drop=True)

    def find_movies_by_title(self, search_text):
        return self.movies_df[
            self.movies_df["title"].str.contains(search_text, case=False, na=False)
        ][["movieId", "title", "genres"]].reset_index(drop=True)

    def get_similar_movies(self, movie_id, n_neighbors=10):
        if movie_id not in self.movie_to_idx:
            return pd.DataFrame(columns=["movieId", "title", "genres", "similarity"])

        movie_idx = self.movie_to_idx[movie_id]

        distances, indices = self.knn_model.kneighbors(
            self.item_user_sparse[movie_idx],
            n_neighbors=min(n_neighbors + 1, len(self.movie_ids)),
        )

        rows = []
        for dist, idx in zip(distances.flatten(), indices.flatten()):
            neighbor_movie_id = self.idx_to_movie[idx]

            if neighbor_movie_id == movie_id:
                continue

            rows.append({"movieId": neighbor_movie_id, "similarity": 1 - dist})

        recs = pd.DataFrame(rows)

        if recs.empty:
            return pd.DataFrame(columns=["movieId", "title", "genres", "similarity"])

        recs = recs.merge(
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        )

        return recs.sort_values("similarity", ascending=False).reset_index(drop=True)

    def build_user_genre_profile(self, user_id):
        user_likes = self.train_df[
            (self.train_df["userId"] == user_id)
            & (self.train_df["rating"] >= self.min_rating_for_profile)
        ][["movieId", "rating"]].copy()

        if user_likes.empty:
            return pd.Series(0.0, index=self.genre_cols)

        liked = user_likes.merge(
            self.genre_df[["movieId"] + self.genre_cols],
            on="movieId",
            how="left",
        ).fillna(0)

        weighted = liked[self.genre_cols].multiply(liked["rating"], axis=0).sum()

        if weighted.max() > 0:
            weighted = weighted / weighted.max()

        return weighted

    def get_genre_score(self, movie_id, user_profile):
        if movie_id not in self.genre_lookup.index:
            return 0.0

        movie_vec = self.genre_lookup.loc[movie_id].astype(float)

        if movie_vec.sum() == 0:
            return 0.0

        return float(np.dot(movie_vec.values, user_profile.values) / movie_vec.sum())

    def recommend_mf(self, user_id, top_n=10):
        if user_id not in self.pred_df.index:
            return self._popularity_fallback(top_n=top_n)

        seen = set(self.train_df.loc[self.train_df["userId"] == user_id, "movieId"])
        scores = self.pred_df.loc[user_id].drop(labels=list(seen), errors="ignore")

        recs = scores.sort_values(ascending=False).head(top_n).reset_index()
        recs.columns = ["movieId", "mf_score"]

        recs = recs.merge(
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        )

        recs = recs.merge(
            self.popularity_df[["movieId", "weighted_score", "rating_count"]],
            on="movieId",
            how="left",
        )

        return recs

    def recommend_cf(self, user_id, top_n=10):
        if user_id not in self.user_to_idx:
            return self._popularity_fallback(top_n=top_n)

        user_history = self.train_df[self.train_df["userId"] == user_id].copy()

        if user_history.empty:
            return self._popularity_fallback(top_n=top_n)

        rated_ids = set(user_history["movieId"])
        candidate_scores = {}

        for _, row in user_history.iterrows():
            seed_movie_id = row["movieId"]
            user_rating = row["rating"]
            similar = self.get_similar_movies(seed_movie_id, self.neighbor_k)

            for _, sim_row in similar.iterrows():
                candidate_id = sim_row["movieId"]
                similarity = sim_row["similarity"]

                if candidate_id in rated_ids:
                    continue
                if similarity < self.min_similarity:
                    continue

                if candidate_id not in candidate_scores:
                    candidate_scores[candidate_id] = {
                        "weighted_sum": 0.0,
                        "similarity_sum": 0.0,
                    }

                candidate_scores[candidate_id]["weighted_sum"] += similarity * user_rating
                candidate_scores[candidate_id]["similarity_sum"] += similarity

        if not candidate_scores:
            return self._popularity_fallback(top_n=top_n, exclude_ids=rated_ids)

        rows = []
        for movie_id, values in candidate_scores.items():
            cf_score = values["weighted_sum"] / values["similarity_sum"]
            rows.append({"movieId": movie_id, "cf_score": cf_score})

        recs = pd.DataFrame(rows)

        recs = recs.merge(
            self.movies_df[["movieId", "title", "genres"]],
            on="movieId",
            how="left",
        )

        recs = recs.merge(
            self.popularity_df[["movieId", "weighted_score", "rating_count"]],
            on="movieId",
            how="left",
        )

        return recs.sort_values(
            ["cf_score", "weighted_score"],
            ascending=False,
        ).head(top_n).reset_index(drop=True)

    def recommend_hybrid(self, user_id, top_n=10):
        if user_id not in self.user_to_idx:
            return self._popularity_fallback(top_n=top_n)

        seen = set(self.train_df.loc[self.train_df["userId"] == user_id, "movieId"])

        cf_recs = self.recommend_cf(user_id, top_n=200)
        mf_recs = self.recommend_mf(user_id, top_n=200)

        candidate_ids = set(cf_recs["movieId"]).union(set(mf_recs["movieId"]))
        candidate_ids = candidate_ids - seen

        if not candidate_ids:
            return self._popularity_fallback(top_n=top_n, exclude_ids=seen)

        cf_lookup = cf_recs.set_index("movieId")["cf_score"].to_dict() if "cf_score" in cf_recs else {}
        mf_lookup = mf_recs.set_index("movieId")["mf_score"].to_dict() if "mf_score" in mf_recs else {}

        user_profile = self.build_user_genre_profile(user_id)
        rows = []

        for movie_id in candidate_ids:
            cf_score = cf_lookup.get(movie_id, 0.0)
            mf_score = mf_lookup.get(movie_id, 0.0)
            genre_score = self.get_genre_score(movie_id, user_profile)

            pop_score = (
                self.popularity_lookup.loc[movie_id, "weighted_score"]
                if movie_id in self.popularity_lookup.index
                else self.global_mean
            )

            rating_count = (
                self.popularity_lookup.loc[movie_id, "rating_count"]
                if movie_id in self.popularity_lookup.index
                else 0
            )

            hybrid_score = (
                self.cf_weight * cf_score
                + self.mf_weight * mf_score
                + self.genre_weight * genre_score
                + self.popularity_weight * (pop_score / 5.0)
            )

            title = self.movie_meta.loc[movie_id, "title"] if movie_id in self.movie_meta.index else None
            genres = self.movie_meta.loc[movie_id, "genres"] if movie_id in self.movie_meta.index else None

            rows.append(
                {
                    "movieId": movie_id,
                    "title": title,
                    "genres": genres,
                    "cf_score": round(cf_score, 4),
                    "mf_score": round(mf_score, 4),
                    "genre_score": round(genre_score, 4),
                    "popularity_score": round(pop_score, 4),
                    "hybrid_score": round(hybrid_score, 4),
                    "rating_count": int(rating_count),
                }
            )

        recs = pd.DataFrame(rows)

        return recs.sort_values(
            ["hybrid_score", "mf_score", "cf_score", "popularity_score"],
            ascending=False,
        ).head(top_n).reset_index(drop=True)

    def recommend_from_custom_ratings(self, custom_ratings_df, top_n=10):
        if custom_ratings_df.empty:
            return self._popularity_fallback(top_n=top_n)

        rated_ids = set(custom_ratings_df["movieId"].astype(int))
        candidate_scores = {}

        for _, row in custom_ratings_df.iterrows():
            seed_movie_id = int(row["movieId"])
            user_rating = float(row["rating"])

            similar = self.get_similar_movies(seed_movie_id, self.neighbor_k)

            for _, sim_row in similar.iterrows():
                candidate_id = int(sim_row["movieId"])
                similarity = float(sim_row["similarity"])

                if candidate_id in rated_ids:
                    continue
                if similarity < self.min_similarity:
                    continue

                if candidate_id not in candidate_scores:
                    candidate_scores[candidate_id] = {
                        "weighted_sum": 0.0,
                        "similarity_sum": 0.0,
                    }

                candidate_scores[candidate_id]["weighted_sum"] += similarity * user_rating
                candidate_scores[candidate_id]["similarity_sum"] += similarity

        if not candidate_scores:
            return self._popularity_fallback(top_n=top_n, exclude_ids=rated_ids)

        rows = []
        for movie_id, values in candidate_scores.items():
            custom_score = values["weighted_sum"] / values["similarity_sum"]
            pop_score = (
                self.popularity_lookup.loc[movie_id, "weighted_score"]
                if movie_id in self.popularity_lookup.index
                else self.global_mean
            )
            final_score = 0.80 * custom_score + 0.20 * (pop_score / 5.0)

            title = self.movie_meta.loc[movie_id, "title"] if movie_id in self.movie_meta.index else None
            genres = self.movie_meta.loc[movie_id, "genres"] if movie_id in self.movie_meta.index else None

            rows.append({
                "movieId": movie_id,
                "title": title,
                "genres": genres,
                "custom_score": round(custom_score, 4),
                "popularity_score": round(pop_score, 4),
                "final_score": round(final_score, 4),
            })

        return (
            pd.DataFrame(rows)
            .sort_values(["final_score", "custom_score", "popularity_score"], ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def explain_recommendation(self, user_id, movie_id):
        if user_id not in self.user_to_idx:
            return {"explanation": "User not found. Recommendation came from popularity fallback."}

        user_history = self.train_df[self.train_df["userId"] == user_id].copy()
        reasons = []

        for _, row in user_history.iterrows():
            seed_movie_id = row["movieId"]
            user_rating = row["rating"]
            similar = self.get_similar_movies(seed_movie_id, self.neighbor_k)
            match = similar[similar["movieId"] == movie_id]

            if not match.empty:
                seed_title = (
                    self.movie_meta.loc[seed_movie_id, "title"]
                    if seed_movie_id in self.movie_meta.index
                    else str(seed_movie_id)
                )

                reasons.append(
                    {
                        "because_you_rated": seed_title,
                        "your_rating": user_rating,
                        "similarity": round(float(match.iloc[0]["similarity"]), 4),
                    }
                )

        if not reasons:
            return {"explanation": "No direct item-neighbor explanation found. Matrix factorization, genre, or popularity likely influenced this recommendation."}

        return pd.DataFrame(reasons).sort_values("similarity", ascending=False).reset_index(drop=True)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def hit_rate_at_k(
    model,
    eval_df,
    top_n=10,
    min_eval_rating=4.0,
    sample_users=100,
    method="hybrid",
):
    eligible_users = []
    train_users = set(model.train_df["userId"].unique())

    for user_id, user_eval in eval_df.groupby("userId"):
        if user_id not in train_users:
            continue
        if (user_eval["rating"] >= min_eval_rating).sum() == 0:
            continue
        eligible_users.append(user_id)

    eligible_users = eligible_users[:sample_users]

    hits = 0
    total = 0

    for user_id in eligible_users:
        actual_positive = set(
            eval_df[
                (eval_df["userId"] == user_id)
                & (eval_df["rating"] >= min_eval_rating)
            ]["movieId"]
        )

        if method == "popularity":
            seen = set(model.train_df.loc[model.train_df["userId"] == user_id, "movieId"])
            recs = model._popularity_fallback(top_n=top_n, exclude_ids=seen)
        elif method == "mf":
            recs = model.recommend_mf(user_id, top_n=top_n)
        elif method == "cf":
            recs = model.recommend_cf(user_id, top_n=top_n)
        else:
            recs = model.recommend_hybrid(user_id, top_n=top_n)

        recommended = set(recs["movieId"])
        hits += int(len(actual_positive.intersection(recommended)) > 0)
        total += 1

    return hits / total if total > 0 else 0.0
