import os
import pandas as pd


class UserProfileStore:
    """
    Lightweight CSV-backed user profile store.

    Stores custom app users separately from MovieLens users.
    """

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if not os.path.exists(path):
            pd.DataFrame(
                columns=["username", "movieId", "title", "rating"]
            ).to_csv(path, index=False)

    def load_all(self):
        return pd.read_csv(self.path)

    def get_users(self):
        df = self.load_all()
        if df.empty:
            return []
        return sorted(df["username"].dropna().unique().tolist())

    def get_user_ratings(self, username):
        df = self.load_all()

        if df.empty or not username:
            return pd.DataFrame(columns=["username", "movieId", "title", "rating"])

        return df[
            df["username"].astype(str).str.lower() == username.lower()
        ].copy()

    def add_or_update_rating(self, username, movie_id, title, rating):
        df = self.load_all()
        username = username.strip()

        if not username:
            raise ValueError("Username cannot be empty.")

        df = df[
            ~(
                (df["username"].astype(str).str.lower() == username.lower())
                & (df["movieId"].astype(int) == int(movie_id))
            )
        ]

        new_row = pd.DataFrame(
            [
                {
                    "username": username,
                    "movieId": int(movie_id),
                    "title": title,
                    "rating": float(rating),
                }
            ]
        )

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(self.path, index=False)

    def delete_rating(self, username, movie_id):
        df = self.load_all()

        df = df[
            ~(
                (df["username"].astype(str).str.lower() == username.lower())
                & (df["movieId"].astype(int) == int(movie_id))
            )
        ]

        df.to_csv(self.path, index=False)

    def clear_user(self, username):
        df = self.load_all()

        df = df[
            df["username"].astype(str).str.lower() != username.lower()
        ]

        df.to_csv(self.path, index=False)
