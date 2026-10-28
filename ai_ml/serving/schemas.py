"""Request / response models for the API. (Bindu)"""
from pydantic import BaseModel


class Recommendation(BaseModel):
    movieId: int
    title: str
    genres: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[Recommendation]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class InfoResponse(BaseModel):
    model_name: str
    metrics: dict
    n_users: int
    min_user_id: int
    max_user_id: int
