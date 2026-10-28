"""Data cleaning. (Dipak)"""
import pandas as pd

from ai_ml.config import LIKE_THRESHOLD


def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows and add the target column `liked` (1 if rating >= 4)."""
    df = df.dropna(subset=["userId", "movieId", "rating"]).drop_duplicates(["userId", "movieId"])
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)].copy()
    df["liked"] = (df["rating"] >= LIKE_THRESHOLD).astype(int)
    return df.reset_index(drop=True)


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the year, and turn 'Action|Comedy' into a list plus a display text."""
    df = df.dropna(subset=["movieId", "title"]).drop_duplicates("movieId").copy()
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("float")
    df["genres_text"] = df["genres"].fillna("").str.replace("|", ", ", regex=False)
    df["genres"] = df["genres"].fillna("").str.split("|")
    return df.reset_index(drop=True)
