"""Train Logistic Regression and Decision Tree, track with MLflow. (Ayush)

Run:  python -m ai_ml.models.train
"""
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ai_ml.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
)
from ai_ml.data.clean import clean_movies, clean_ratings
from ai_ml.data.features import (
    build_features,
    build_genre_matrix,
    compute_stats,
    feature_columns,
    get_genres,
)
from ai_ml.data.load import load_movies, load_ratings
from ai_ml.models.evaluate import evaluate

DEFAULT_PARAMS = {
    "logistic_regression": {"C": 1.0},
    "decision_tree": {"max_depth": 8, "min_samples_leaf": 50, "criterion": "gini"},
}


def build_model(name: str, params: dict):
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **params)
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Unknown model: {name}")


def prepare_data(test_size: float = 0.2) -> dict:
    """Load, clean, split and build features. Stats come from the train split only."""
    ratings = clean_ratings(load_ratings())
    movies = clean_movies(load_movies())
    train, test = train_test_split(
        ratings, test_size=test_size, random_state=RANDOM_STATE, stratify=ratings["liked"]
    )
    user_stats, movie_stats, global_mean = compute_stats(train)
    genre_matrix = build_genre_matrix(movies, get_genres(movies))
    args = (user_stats, movie_stats, genre_matrix, global_mean)
    return {
        "X_train": build_features(train, *args),
        "y_train": train["liked"],
        "X_test": build_features(test, *args),
        "y_test": test["liked"],
        "user_stats": user_stats,
        "movie_stats": movie_stats,
        "global_mean": global_mean,
        "genre_matrix": genre_matrix,
        "movies": movies,
        "user_seen": ratings.groupby("userId")["movieId"].apply(set).to_dict(),
    }


def fit_and_log(name: str, params: dict, data: dict, stage: str = "baseline"):
    """Train one model, log params/metrics/model to MLflow, return (model, metrics)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{name}-{stage}"):
        model = build_model(name, params)
        model.fit(data["X_train"], data["y_train"])
        metrics = evaluate(model, data["X_test"], data["y_test"])
        mlflow.set_tag("stage", stage)
        mlflow.log_param("model", name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
    return model, metrics


def save_bundle(model, name: str, metrics: dict, data: dict) -> None:
    """Save the model together with everything the API needs to make recommendations."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "model_name": name,
        "metrics": metrics,
        "feature_columns": feature_columns(data["genre_matrix"]),
        "user_stats": data["user_stats"],
        "movie_stats": data["movie_stats"],
        "global_mean": data["global_mean"],
        "genre_matrix": data["genre_matrix"],
        "movies": data["movies"],
        "user_seen": data["user_seen"],
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved best model ({name}) to {MODEL_PATH}")


def run_experiments(params_by_model: dict, stage: str, data: dict | None = None) -> str:
    """Train every model, print a comparison, save the one with the best ROC-AUC."""
    data = data or prepare_data()
    results = {}
    for name, params in params_by_model.items():
        model, metrics = fit_and_log(name, params, data, stage)
        results[name] = (model, metrics)
        print(f"{name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    best = max(results, key=lambda n: results[n][1]["roc_auc"])
    save_bundle(results[best][0], best, results[best][1], data)
    return best


def main() -> None:
    run_experiments(DEFAULT_PARAMS, stage="baseline")


if __name__ == "__main__":
    main()
