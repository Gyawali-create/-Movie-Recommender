import pytest
from fastapi.testclient import TestClient

from ai_ml.config import MODEL_PATH
from api.main import app

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first")


def test_health_works_even_without_model():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@needs_model
def test_recommend_returns_top_n():
    with TestClient(app) as client:
        user_id = client.get("/info").json()["min_user_id"]
        resp = client.get(f"/recommend/{user_id}", params={"top_n": 5})
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) == 5


@needs_model
def test_unknown_user_returns_404():
    with TestClient(app) as client:
        assert client.get("/recommend/99999999").status_code == 404
