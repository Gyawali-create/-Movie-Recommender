"""Model evaluation and cost function. (Ayush)"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def recommendation_cost(y_true, y_pred, fp_cost: float = 1.0, fn_cost: float = 0.5) -> float:
    """Average cost per prediction (lower is better).

    False positive = we recommend a movie the user dislikes (bad experience, cost 1.0).
    False negative = we miss a movie the user would like (lost chance, cost 0.5).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return float((fp * fp_cost + fn * fn_cost) / len(y_true))


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "cost": recommendation_cost(y, pred),
    }
