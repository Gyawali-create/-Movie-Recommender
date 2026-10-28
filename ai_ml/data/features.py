"""Feature engineering. (Dipak)

Statistics (user/movie averages) are computed on the TRAIN split only,
so the test set stays honest (no data leakage from test ratings).
"""
import pandas as pd

BASE_FEATURES = ["user_mean", "user_count", "movie_mean", "movie_count"]


def get_genres(movies: pd.DataFrame) -> list[str]:
    found: set[str] = set()
    for genre_list in movies["genres"]:
        found.update(genre_list)
    found.discard("(no genres listed)")
    found.discard("")
    return sorted(found)


def build_genre_matrix(movies: pd.DataFrame, genres: list[str]) -> pd.DataFrame:
    """One 0/1 column per genre, one row per movie."""
    matrix = pd.DataFrame({"movieId": movies["movieId"].values})
    for genre in genres:
        matrix[f"genre_{genre}"] = [int(genre in lst) for lst in movies["genres"]]
    return matrix


def compute_stats(train: pd.DataFrame):
    """Average rating and number of ratings per user and per movie."""
    user_stats = (
        train.groupby("userId")["rating"].agg(user_mean="mean", user_count="count").reset_index()
    )
    movie_stats = (
        train.groupby("movieId")["rating"].agg(movie_mean="mean", movie_count="count").reset_index()
    )
    return user_stats, movie_stats, float(train["rating"].mean())


def feature_columns(genre_matrix: pd.DataFrame) -> list[str]:
    return BASE_FEATURES + [c for c in genre_matrix.columns if c != "movieId"]


def build_features(pairs, user_stats, movie_stats, genre_matrix, global_mean) -> pd.DataFrame:
    """Build the model input for (userId, movieId) pairs."""
    df = pairs[["userId", "movieId"]].copy()
    df = df.merge(user_stats, on="userId", how="left")
    df = df.merge(movie_stats, on="movieId", how="left")
    df = df.merge(genre_matrix, on="movieId", how="left")
    df["user_mean"] = df["user_mean"].fillna(global_mean)
    df["movie_mean"] = df["movie_mean"].fillna(global_mean)
    df["user_count"] = df["user_count"].fillna(0)
    df["movie_count"] = df["movie_count"].fillna(0)
    cols = feature_columns(genre_matrix)
    df[cols] = df[cols].fillna(0)
    df.index = pairs.index
    return df[cols]
