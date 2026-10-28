"""Download and load the MovieLens dataset. (Dipak)"""
import io
import urllib.error
import urllib.request
import zipfile

import pandas as pd

from ai_ml.config import DATA_RAW, MOVIELENS_DIR, MOVIELENS_URL

MANUAL_HELP = f"""
Could not download the dataset (network or SSL problem).
Do it manually:
  1. Open {MOVIELENS_URL} in your browser
  2. Extract the zip so these files exist:
       {MOVIELENS_DIR}\\ratings.csv
       {MOVIELENS_DIR}\\movies.csv
Or, only to test the project with DUMMY data:  python -m ai_ml.data.sample
"""


def download_movielens(force: bool = False) -> None:
    """Download ml-latest-small into data/raw/ (skipped if already there)."""
    if MOVIELENS_DIR.exists() and not force:
        print(f"Dataset already exists at {MOVIELENS_DIR}")
        return
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MOVIELENS_URL} ...")
    try:
        with urllib.request.urlopen(MOVIELENS_URL, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError):
        print(MANUAL_HELP)
        raise SystemExit(1)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DATA_RAW)
    print(f"Saved to {MOVIELENS_DIR}")


def _read(name: str) -> pd.DataFrame:
    path = MOVIELENS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run  python -m ai_ml.data.load  (real data)  or  python -m ai_ml.data.sample  (dummy data)."
        )
    return pd.read_csv(path)


def load_ratings() -> pd.DataFrame:
    return _read("ratings.csv")


def load_movies() -> pd.DataFrame:
    return _read("movies.csv")


if __name__ == "__main__":
    download_movielens()
