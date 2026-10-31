"""Hyperparameter tuning with  Optuna. (Ayush)

Run:  python -m ai_ml.models.tune --trials 30
"""
import argparse

import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ai_ml.config import RANDOM_STATE
from ai_ml.models.train import build_model, prepare_data, run_experiments


def suggest_params(trial: optuna.Trial, name: str) -> dict:
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
    }


def tune_model(name: str, data: dict, n_trials: int) -> dict:
    """Search the best parameters using a validation split of the TRAIN data."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        data["X_train"], data["y_train"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=data["y_train"],
    )

    def objective(trial: optuna.Trial) -> float:
        model = build_model(name, suggest_params(trial, name))
        model.fit(X_tr, y_tr)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(objective, n_trials=n_trials)
    print(f"{name}: best validation ROC-AUC={study.best_value:.4f} params={study.best_params}")
    return study.best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = prepare_data()
    best = {name: tune_model(name, data, args.trials)
            for name in ("logistic_regression", "decision_tree")}
    run_experiments(best, stage="tuned", data=data)


if __name__ == "__main__":
    main()
