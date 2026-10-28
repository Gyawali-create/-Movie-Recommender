# 🎬 Movie Recommender (AI/ML Team Project)

A movie recommendation system built on the **MovieLens** dataset. We train two models
(**Logistic Regression** and **Decision Tree**) to predict whether a user will like a movie
(rating ≥ 4), tune them with **Optuna**, track every experiment in **MLflow**, serve the best
model through **FastAPI**, and show recommendations in a **Streamlit** app.

**Flow:** `Data → Features → Train → Track (MLflow) → Save model → FastAPI → Streamlit`

---

## 1. Team and responsibilities

| Person | Part | Main files |
|---|---|---|
| **Dipak** | Data: explore, clean, features, data versioning (DVC) | `ai_ml/data/`, `notebooks/01_explore_data.ipynb` |
| **Ayush** | Models: train, evaluate (cost function), Optuna tuning, MLflow | `ai_ml/models/` |
| **Bindu** | Project setup, Git and branching, docs and sheets, FastAPI, Streamlit, tests | `api/`, `ui/`, `ai_ml/serving/`, `docs/`, `tests/`, `.github/` |

Every person makes **their own commits** and **reviews at least one teammate's pull request**.

---

## 2. Tools and why we use them

| Tool | What it is | Why we use it |
|---|---|---|
| Python | Programming language | Everything is written in it |
| MovieLens (ml-latest-small) | Public dataset of users, movies and ratings | Data to learn from |
| Pandas / NumPy | Data libraries | Clean and prepare data |
| Jupyter Notebook | Interactive code document | Explore data, show findings |
| DVC | Data Version Control (Git for data files) | Know which data version a model used |
| scikit-learn | ML library | Logistic Regression and Decision Tree |
| Evaluation + cost function | Accuracy, F1, ROC-AUC and a custom cost | Decide which model is better |
| Optuna | Hyperparameter tuning library | Finds the best model settings automatically |
| MLflow | Experiment tracking | Records settings, scores and models of every run |
| Aim | Another tracker (optional) | Only add if the mentor asks. MLflow is enough |
| joblib (`model.pkl`) | Saves scikit-learn models | Load the model later without retraining (`.pt` is only for PyTorch) |
| FastAPI | API framework | Backend: request in, recommendations out |
| Streamlit | Web app framework | Frontend (`ui/app.py`) that calls the API |
| Git + GitHub (public) | Version control and hosting | Teamwork, pull requests, reviews |

---

## 3. Folder structure

```
movie-recommender/
├── .github/pull_request_template.md   PR checklist
├── ai_ml/                             the Python package (all project code)
│   ├── config.py                      paths and settings
│   ├── data/      load.py, sample.py, clean.py, features.py     (Dipak)
│   ├── models/    train.py, evaluate.py, tune.py                (Ayush)
│   └── serving/   recommender.py, schemas.py                    (Bindu)
├── api/main.py                        FastAPI app
├── ui/app.py                          Streamlit app (calls the API)
├── data/
│   ├── raw/                           dataset (DVC tracked, not in Git)
│   └── processed/
├── docs/
│   ├── standup_sheet.md               daily standup
│   ├── feedback_sheet.md              peer feedback + elevator pitch
│   ├── branching_and_commits.md       Git rules
│   └── findings.docx                  final report (add at the end)
├── models/                            model.pkl is saved here (not in Git)
├── notebooks/                         01_explore_data.ipynb, 02_model_training.ipynb
├── tests/                             test_data.py, test_models.py, test_api.py
├── setup.bat  train.bat  run_api.bat  run_ui.bat  run_mlflow.bat  run_tests.bat
├── requirements.txt                   all dependencies
├── pytest.ini
├── .env.example                       copy to .env (never commit .env)
├── .gitignore
├── CONTRIBUTING.md
└── README.md
```

---

## 4. Setup (Windows, one time)

Put the project in a normal folder such as `D:\movie-recommender`, then double-click or run:

```cmd
setup.bat
```

It creates `.venv` (Python 3.12 if available), installs `requirements.txt` and creates `.env`.
It already contains the pip flags that avoid the "CERTIFICATE_VERIFY_FAILED" error on networks that scan HTTPS.

Manual version (Command Prompt):
```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy .env.example .env
```
In PowerShell use `.venv\Scripts\Activate.ps1` instead of `activate.bat`.

## 5. How to run

| Step | Double-click / run | Manual command (venv activated, from the project folder) |
|---|---|---|
| 1. Get data | (done inside `train.bat`) | `python -m ai_ml.data.load` |
| 2. Train and save model | `train.bat` | `python -m ai_ml.models.train` |
| 3. Tune with Optuna (optional) | | `python -m ai_ml.models.tune --trials 30` |
| 4. Compare runs in MLflow | `run_mlflow.bat` | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| 5. Start the API (keep it open) | `run_api.bat` | `uvicorn api.main:app --reload` |
| 6. Start the app (second window) | `run_ui.bat` | `streamlit run ui/app.py` |
| Tests | `run_tests.bat` | `python -m pytest` |

API docs: http://127.0.0.1:8000/docs, MLflow: http://127.0.0.1:5000

**If the dataset download fails** (SSL or network problem): download
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip in your browser and extract it so that
`data/raw/ml-latest-small/ratings.csv` and `movies.csv` exist.

**Just testing?** `python -m ai_ml.data.sample` creates DUMMY data so you can check that everything runs.
Never use dummy-data results in your findings.

---

## 6. How it works

1. **Data (Dipak):** ratings are cleaned and each rating gets a label `liked = 1` if rating ≥ 4.
2. **Features (Dipak):** average rating and rating count per user and per movie, plus one 0/1 column per genre.
   These statistics are computed on the **train split only** to avoid data leakage.
3. **Models (Ayush):** Logistic Regression and Decision Tree are trained and compared with
   accuracy, precision, recall, F1, ROC-AUC and a **cost function**
   (a disliked recommendation costs 1.0, a missed good movie costs 0.5). Best ROC-AUC wins.
4. **Tracking (Ayush):** every run logs parameters, metrics and the model to MLflow.
5. **Serving (Bindu):** for a user, the API scores every movie they have not seen and returns the top N.
6. **UI (Bindu):** Streamlit asks the API and shows the table.

### API endpoints

| Method | URL | Returns |
|---|---|---|
| GET | `/health` | API status and whether the model is loaded |
| GET | `/info` | model name, metrics, valid user ID range |
| GET | `/recommend/{user_id}?top_n=10` | top N movies with a score |

## 7. Data versioning with DVC (Dipak)

```powershell
dvc init
dvc add data/raw/ml-latest-small
git add data/raw/ml-latest-small.dvc data/raw/.gitignore .dvc
git commit -m "chore: track dataset with DVC"
```
Optional: `dvc remote add -d storage <folder-or-cloud-path>` then `dvc push`.

---

## 8. Git workflow

**Branches:** `main` (final, stable) ← `test` (checking) ← `dev` (daily work) ← `feature/*` (one per task)

```powershell
git init
git branch -M main
git add .
git commit -m "chore: init repo with folder structure"
git branch dev
git branch test
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main dev test
```

Daily work:
```powershell
git checkout dev
git pull
git checkout -b feature/model-training
# ... work ...
git add <files>
git commit -m "feat: add decision tree model"
git push -u origin feature/model-training
# open a Pull Request into dev, a teammate reviews, then merge
```

Each person sets their own identity so commits show the right author:
```powershell
git config user.name "Your Name"
git config user.email "you@example.com"
```
Check individual commits: `git log --author="Ayush" --oneline`

**Never commit:** `__pycache__/`, `.env`, `mlruns/`, `mlflow.db`, `data/raw/*`, `models/*.pkl`
(all already in `.gitignore`).

### Commit message convention
`type: short message` with types `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
Examples: `feat: add FastAPI /recommend endpoint`, `fix: handle unknown user id`.

### Planned commits per person

**Dipak**
1. `docs: add dataset exploration notebook`
2. `feat: add data loading in ai_ml/data/load.py`
3. `feat: add data cleaning in clean.py`
4. `feat: add feature engineering in features.py`
5. `chore: track dataset with DVC`
6. `test: add data tests`

**Ayush**
1. `feat: add baseline logistic regression and decision tree in train.py`
2. `feat: add evaluation metrics and cost function in evaluate.py`
3. `feat: integrate MLflow tracking`
4. `feat: add Optuna tuning in tune.py`
5. `chore: save best model bundle to models/`
6. `test: add model tests`

**Bindu**
1. `chore: init repo with folder structure and .gitkeep files`
2. `chore: add .gitignore and .env.example`
3. `chore: add dependencies in requirements.txt and setup scripts`
4. `chore: create main, dev and test branches`
5. `docs: add README, CONTRIBUTING and branching rules`
6. `docs: add standup and feedback sheets`
7. `chore: add pull request template`
8. `feat: add config.py, recommender.py and schemas.py`
9. `feat: add FastAPI app with /health, /info and /recommend`
10. `feat: add Streamlit app that calls the API`
11. `test: add API tests`
12. `docs: add run instructions and screenshots`

Bindu pushes commits 1 to 7 to `dev` first so the others can start.

---

## 9. Team process

- **Standup:** fill `docs/standup_sheet.md` daily (yesterday done, today to-do, blockers).
- **Review:** every pull request needs one review from a teammate.
- **Feedback:** fill `docs/feedback_sheet.md` for each teammate at the end.
- **Findings:** write results into `docs/findings.docx` (dataset insights, model comparison table,
  MLflow screenshots, best model, API and Streamlit screenshots).

## 10. Known limitations (good points for the findings)

- Averages for training rows include the row's own rating, so training scores look slightly better than test scores.
- Recommendations only use user and movie averages and genres, not "users similar to you".
- New users (not in the dataset) are not supported yet.
- Possible next steps: collaborative filtering (matrix factorization), more features (year, tags), a model comparison with Random Forest.

## 11. Elevator pitch

> "We built a movie recommender using the MovieLens dataset. We trained Logistic Regression and a
> Decision Tree with scikit-learn, tuned them with Optuna and tracked experiments in MLflow.
> The best model is served through FastAPI and a Streamlit app calls that API to show
> recommendations. We worked on GitHub with main, dev and test branches, and each member has
> their own commits and reviews."
#   - M o v i e - R e c o m m e n d e r  
 