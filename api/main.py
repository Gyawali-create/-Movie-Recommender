"""FastAPI backend. (Bindu)

Run:  uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request

from ai_ml.serving.recommender import Recommender, UnknownUserError
from ai_ml.serving.schemas import HealthResponse, InfoResponse, RecommendResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.recommender = Recommender()
    except FileNotFoundError:
        app.state.recommender = None  # model not trained yet
    yield


app = FastAPI(title="Movie Recommender API", version="0.1.0", lifespan=lifespan)


def get_recommender(request: Request) -> Recommender:
    rec = request.app.state.recommender
    if rec is None:
        raise HTTPException(503, "Model not found. Run: python -m ai_ml.models.train")
    return rec


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.recommender is not None}


@app.get("/info", response_model=InfoResponse)
def info(request: Request):
    return get_recommender(request).info()


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(request: Request, user_id: int, top_n: int = Query(10, ge=1, le=50)):
    rec = get_recommender(request)
    try:
        items = rec.recommend(user_id, top_n)
    except UnknownUserError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"user_id": user_id, "recommendations": items}
