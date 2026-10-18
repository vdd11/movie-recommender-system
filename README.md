# FlikPik AI — AI Entertainment Discovery Platform

> A hybrid recommendation engine that helps people decide what to watch next — combining collaborative filtering, AI-powered search, trending signals, and trailer integration in one app.

**Live demo:** [flikpik-ai.onrender.com](https://flikpik-ai.onrender.com/)

**My role:** Data Engineer & Data Prep Lead on a 4-person capstone team (see [My Role](#my-role)).

**Tech stack:** Python, pandas, NumPy, scikit-learn, SciPy, Streamlit, TMDB API, Render

![FlikPik AI home page with trending titles](screenshots/home-trending.png)

---

## Problem

Choosing what to watch is harder than it should be — recommendations are scattered across streaming services, and most recommenders rely on a single signal. FlikPik AI brings hybrid recommendations, semantic search, social/trending signals, trailers, and streaming availability into one place, so a user can go from "I don't know what to watch" to a confident pick in a couple of clicks.

## My Role

Built as a 4-person capstone project for the 4Geeks Academy Data Science & Machine Learning program. I served as **Data Engineer & Data Prep Lead**, responsible for:

- Designing and implementing the time-based train / validation / test split methodology to prevent data leakage
- Constructing the sparse 610 × 9,724 user-item matrix (98.3% sparsity) that the modeling team built on
- Data cleaning and exploratory data analysis on the MovieLens ratings dataset
- Feature engineering (average rating, rating count, Bayesian popularity score, recency)
- Validating referential integrity across all four source tables
- Delivering six clean, validated output files to teammates and aligning on data schemas

## Dataset

- **Source:** [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) (`ml-latest-small`) — a standard ML research benchmark maintained by the GroupLens group at the University of Minnesota
- **Size:** 100,836 ratings across 9,742 movies and 610 users
- **Enrichment:** TMDB API integration for real-time movie metadata, posters, and trailers
- **Key features:** user IDs, movie IDs, ratings (0.5–5.0), timestamps, one-hot encoded genres, plus TMDB-sourced metadata

## Approach

The pipeline runs from raw ratings through to a deployed app:

1. **Data cleaning & EDA** — handled missing values, verified zero duplicates, validated referential integrity across all tables, converted Unix timestamps to datetime, and one-hot encoded pipe-separated genre strings into 20 binary columns.
2. **Feature engineering** — built average rating, rating count, Bayesian popularity score (balancing quality against sample size), and recency features per movie. Also computed user-level rating stats.
3. **Train/validation/test split** — time-based 80/10/10 split sorted by timestamp so the model trains on historical ratings and evaluates on future ones, preventing data leakage.
4. **Sparse matrix construction** — built a 610 × 9,724 user-item matrix with 98.3% sparsity, stored as a CSR sparse matrix for memory efficiency.
5. **Modeling** — hybrid recommender combining item-item collaborative filtering (CF), matrix factorization via TruncatedSVD (50 latent factors), genre-based content filtering, and a popularity baseline. Weighted ensemble: CF 30%, MF 45%, Genre 15%, Popularity 10%.
6. **Deployment** — packaged with a Streamlit interface and deployed live on Render.

## Results

The deployed app returns personalized hybrid recommendations, AI-powered natural language search results, trending titles, and trailers for any title in the catalog.

Offline evaluation (`notebooks/04_evaluation_comparison.ipynb`) uses the time-based split, ranks the top 10 recommendations per user, and counts a movie as relevant if the user rated it 4.0 or higher in the evaluation split:

| Model | Split | Hit Rate@10 | Precision@10 | Recall@10 | Catalog coverage@10 |
|---|---|---|---|---|---|
| Popularity | Validation | 0.19 | 0.026 | 0.037 | 0.4% |
| Matrix Factorization | Validation | 0.34 | 0.039 | 0.079 | 3.3% |
| Hybrid | Validation | 0.35 | 0.042 | 0.084 | 4.5% |
| Popularity | Test | 0.18 | 0.021 | 0.034 | 0.4% |
| Matrix Factorization | Test | 0.27 | 0.037 | 0.071 | 3.3% |
| Hybrid | Test | 0.27 | 0.038 | 0.065 | 4.5% |

- Matrix factorization and the hybrid model both clearly beat the popularity baseline on every metric, and the hybrid model exposes more of the catalog (4.5% vs. 0.4% for popularity).
- The hybrid model is roughly tied with matrix factorization on hit rate, so these results do not show a clear accuracy gain from the ensemble. On the test split its recall (0.065) is slightly lower than matrix factorization's (0.071).
- **Caveats:** each number uses only the first 100 eligible users (by user ID) in the split, so differences of a few points are within noise. Results come from a single run. The notebook also defines an item-item KNN baseline that was not included in this table.

## Features

**Discovery**
- Hybrid recommendation engine (CF + MF + Genre + Popularity)
- AI-powered semantic search ("Ask FlikPik")
- Similar movie finder
- Actor / Director search

**Social & Trending**
- Trending intelligence — what's popular this week
- Social buzz signals

**Rich Media**
- Real movie posters and trailers via TMDB
- Streaming availability (Netflix, Disney+, Hulu, and more)

**Personalization**
- Personalized taste insights
- User rating system

## Run It Locally

```bash
git clone https://github.com/vdd11/movie-recommender-system.git
cd movie-recommender-system
pip install -r requirements.txt
```

FlikPik AI needs a TMDB API key to fetch posters and metadata. Get a free key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api), then create a `.env` file in the project root (it is git-ignored; never commit it). A `YOUTUBE_API_KEY` is optional and enables automatic trailer lookup.

```
TMDB_API_KEY=your_key_here
YOUTUBE_API_KEY=optional_key_here
```

Train the recommender (writes `models/hybrid_recommender.pkl`; the app will also train one on first launch if it is missing), then start the app from the project root:

```bash
python src/train_model.py
streamlit run app/app.py
```

`app/app.py` is the entry point used by both the `Dockerfile` and `render.yaml`.

## Screenshots

![Actor / director discovery for Quentin Tarantino](screenshots/actor-director-search.png)
![Search results for "Lord of the Rings"](screenshots/title-search-results.png)

## What I'd Do Next

- **Scale the dataset** — upgrade from MovieLens Latest Small (about 100K ratings) to the full MovieLens 25M dataset for richer, more personalized recommendations with significantly reduced sparsity.
- **Polish the frontend** — tighten the UI, fix any rendering issues (a few score cards were displaying raw HTML in testing), and improve mobile responsiveness.
- **Add cold-start handling** — build a proper onboarding flow where new users rate a handful of movies before receiving personalized recommendations, rather than falling back to the popularity baseline.
- **Cache TMDB responses** — reduce external API calls and improve load times by caching poster and metadata responses locally.

## Team

A 4-person 4Geeks Academy capstone project.

| Name | GitHub |
|------|--------|
| Matthew Kane | [matthewkane-ml](https://github.com/matthewkane-ml) |
| Tomeka L. German | [tomekagerman](https://github.com/tomekagerman) |
| Vishal Desai | [vdd11](https://github.com/vdd11) |
| Brandon Henry | [bhenrywc](https://github.com/bhenrywc) |
