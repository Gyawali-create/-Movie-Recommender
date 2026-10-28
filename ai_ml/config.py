"""Central place for paths and settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_DIR = DATA_RAW / "ml-latest-small"

LIKE_THRESHOLD = 4.0  # rating >= 4.0 means the user "liked" the movie
RANDOM_STATE = 42

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "movie-recommender"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
