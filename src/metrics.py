import numpy as np
import pandas as pd
from recommender import hit_rate_at_k


def evaluate_ranking_metrics(
    model,
    eval_df,
    top_n=10,
    min_eval_rating=4.0,
    sample_users=100,
    method="hybrid",
):
    eligible_users = []
    train_user_set = set(model.train_df["userId"].unique())

    for user_id, user_eval in eval_df.groupby("userId"):
        relevant = set(user_eval.loc[user_eval["rating"] >= min_eval_rating, "movieId"].tolist())

        if not relevant:
            continue

        if user_id not in train_user_set:
            continue

        eligible_users.append(user_id)

    eligible_users = eligible_users[:sample_users]

    precision_scores = []
    recall_scores = []
    all_recommended_items = set()

    for user_id in eligible_users:
        relevant_items = set(
            eval_df[
                (eval_df["userId"] == user_id)
                & (eval_df["rating"] >= min_eval_rating)
            ]["movieId"].tolist()
        )

        if method == "popularity":
            seen_items = set(model.train_df.loc[model.train_df["userId"] == user_id, "movieId"].tolist())
            recs = model._popularity_fallback(top_n=top_n, exclude_ids=seen_items)
        elif method == "mf":
            recs = model.recommend_mf(user_id=user_id, top_n=top_n)
        elif method == "cf":
            recs = model.recommend_cf(user_id=user_id, top_n=top_n)
        else:
            recs = model.recommend_hybrid(user_id=user_id, top_n=top_n)

        recommended_items = set(recs["movieId"].tolist())
        all_recommended_items.update(recommended_items)

        hits = len(recommended_items.intersection(relevant_items))

        precision_scores.append(hits / top_n if top_n > 0 else 0.0)
        recall_scores.append(hits / len(relevant_items) if relevant_items else 0.0)

    catalog_size = len(set(model.movies_df["movieId"].tolist()))
    coverage = len(all_recommended_items) / catalog_size if catalog_size > 0 else 0.0

    return {
        "precision@k": float(np.mean(precision_scores)) if precision_scores else 0.0,
        "recall@k": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "coverage@k": float(coverage),
        "users_evaluated": len(precision_scores),
    }


def compare_models(
    model,
    val_df,
    test_df,
    top_n=10,
    min_eval_rating=4.0,
    sample_users=100,
):
    rows = []

    for split_name, split_df in [("Validation", val_df), ("Test", test_df)]:
        for method, label in [
            ("popularity", "Popularity"),
            ("cf", "Item-Item CF"),
            ("mf", "Matrix Factorization"),
            ("hybrid", "Upgraded Hybrid"),
        ]:
            hit_rate = hit_rate_at_k(
                model=model,
                eval_df=split_df,
                top_n=top_n,
                min_eval_rating=min_eval_rating,
                sample_users=sample_users,
                method=method,
            )

            ranking_metrics = evaluate_ranking_metrics(
                model=model,
                eval_df=split_df,
                top_n=top_n,
                min_eval_rating=min_eval_rating,
                sample_users=sample_users,
                method=method,
            )

            rows.append(
                {
                    "split": split_name,
                    "model": label,
                    "hit_rate@k": hit_rate,
                    **ranking_metrics,
                }
            )

    return pd.DataFrame(rows)
