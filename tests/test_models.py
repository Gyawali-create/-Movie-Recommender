import numpy as np
import pytest

from ai_ml.models.evaluate import evaluate, recommendation_cost
from ai_ml.models.train import DEFAULT_PARAMS, build_model


def test_cost_function():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 1, 0, 0]  # one false positive (1.0) + one false negative (0.5)
    assert recommendation_cost(y_true, y_pred) == pytest.approx(1.5 / 4)


@pytest.mark.parametrize("name", ["logistic_regression", "decision_tree"])
def test_models_train_and_evaluate(name):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=200) > 0).astype(int)
    model = build_model(name, DEFAULT_PARAMS[name])
    model.fit(X, y)
    metrics = evaluate(model, X, y)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc", "cost"}
    assert metrics["roc_auc"] > 0.8
