"""Load the saved model and produce top-N recommendations. (Bindu)"""
import joblib

from ai_ml.config import MODEL_PATH
from ai_ml.data.features import build_features


class UnknownUserError(ValueError):
    pass


class Recommender:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.model_name = bundle["model_name"]
        self.metrics = bundle["metrics"]
        self.user_stats = bundle["user_stats"]
        self.movie_stats = bundle["movie_stats"]
        self.global_mean = bundle["global_mean"]
        self.genre_matrix = bundle["genre_matrix"]
        self.movies = bundle["movies"]
        self.user_seen = bundle["user_seen"]
        self.known_users = set(self.user_stats["userId"].tolist())

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "n_users": len(self.known_users),
            "min_user_id": int(min(self.known_users)),
            "max_user_id": int(max(self.known_users)),
        }

    def recommend(self, user_id: int, top_n: int = 10, min_votes: int = 5) -> list[dict]:
        if user_id not in self.known_users:
            raise UnknownUserError(f"User {user_id} not found")

        seen = self.user_seen.get(user_id, set())
        stats = self.movie_stats
        cand = stats[(stats["movie_count"] >= min_votes) & (~stats["movieId"].isin(seen))]
        cand = cand[["movieId"]].copy()
        cand["userId"] = user_id

        X = build_features(cand, self.user_stats, self.movie_stats,
                           self.genre_matrix, self.global_mean)
        cand["score"] = self.model.predict_proba(X)[:, 1]
        top = cand.nlargest(top_n, "score").merge(
            self.movies[["movieId", "title", "genres_text"]], on="movieId", how="left"
        )
        return [
            {"movieId": int(r.movieId), "title": r.title,
             "genres": r.genres_text, "score": round(float(r.score), 4)}
            for r in top.itertuples()
        ]
