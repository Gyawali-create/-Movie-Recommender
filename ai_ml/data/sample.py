"""Create DUMMY MovieLens-style data, only to test that the project runs. (Dipak)

Run:  python -m ai_ml.data.sample
Do NOT use these results in your findings. Use the real MovieLens data for that.
"""
import numpy as np
import pandas as pd

from ai_ml.config import MOVIELENS_DIR

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]


def make_sample(n_users=300, n_movies=800, per_user=80, seed=42) -> None:
    rng = np.random.default_rng(seed)
    movie_genres = [list(rng.choice(GENRES, size=rng.integers(1, 4), replace=False))
                    for _ in range(n_movies)]
    movies = pd.DataFrame({
        "movieId": np.arange(1, n_movies + 1),
        "title": [f"Sample Movie {i} ({1960 + i % 60})" for i in range(1, n_movies + 1)],
        "genres": ["|".join(g) for g in movie_genres],
    })

    quality = rng.normal(0, 0.6, n_movies)
    user_bias = rng.normal(0, 0.5, n_users)
    user_pref = rng.normal(0, 0.8, (n_users, len(GENRES)))
    popularity = rng.pareto(1.5, n_movies) + 1
    popularity /= popularity.sum()

    rows = []
    for u in range(n_users):
        for m in rng.choice(n_movies, size=per_user, replace=False, p=popularity):
            g_idx = [GENRES.index(g) for g in movie_genres[m]]
            raw = 3.5 + user_bias[u] + quality[m] + 0.4 * user_pref[u, g_idx].sum() + rng.normal(0, 0.6)
            rows.append((u + 1, m + 1, float(np.clip(round(raw * 2) / 2, 0.5, 5.0)), 0))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])

    MOVIELENS_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(MOVIELENS_DIR / "ratings.csv", index=False)
    movies.to_csv(MOVIELENS_DIR / "movies.csv", index=False)
    print(f"DUMMY data written to {MOVIELENS_DIR} ({len(ratings)} ratings).")
    print("Use it only to test the project. For real results use the MovieLens download.")


if __name__ == "__main__":
    make_sample()
