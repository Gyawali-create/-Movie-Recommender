import pandas as pd

from ai_ml.data.clean import clean_movies, clean_ratings
from ai_ml.data.features import build_features, build_genre_matrix, compute_stats, get_genres


def sample():
    ratings = pd.DataFrame({
        "userId": [1, 1, 2, 2, 2],
        "movieId": [10, 20, 10, 20, 20],  # last row is a duplicate of (2, 20)
        "rating": [5.0, 2.0, 4.0, 3.5, 3.5],
    })
    movies = pd.DataFrame({
        "movieId": [10, 20],
        "title": ["Alpha (1999)", "Beta (2005)"],
        "genres": ["Action|Comedy", "(no genres listed)"],
    })
    return ratings, movies


def test_clean_ratings_adds_label_and_drops_duplicates():
    ratings, _ = sample()
    out = clean_ratings(ratings)
    assert len(out) == 4
    assert out.loc[(out.userId == 1) & (out.movieId == 10), "liked"].item() == 1
    assert out.loc[(out.userId == 1) & (out.movieId == 20), "liked"].item() == 0


def test_clean_movies_extracts_year_and_genres():
    _, movies = sample()
    out = clean_movies(movies)
    assert out["year"].tolist() == [1999.0, 2005.0]
    assert out.loc[0, "genres"] == ["Action", "Comedy"]
    assert out.loc[0, "genres_text"] == "Action, Comedy"


def test_features_have_no_missing_values():
    ratings, movies = sample()
    ratings, movies = clean_ratings(ratings), clean_movies(movies)
    gm = build_genre_matrix(movies, get_genres(movies))
    user_stats, movie_stats, mean = compute_stats(ratings)
    pairs = pd.DataFrame({"userId": [1, 99], "movieId": [10, 777]})  # unseen user/movie
    X = build_features(pairs, user_stats, movie_stats, gm, mean)
    assert not X.isna().any().any()
    assert X.loc[1, "user_mean"] == mean
