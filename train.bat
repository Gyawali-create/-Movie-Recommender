@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if not exist data\raw\ml-latest-small\ratings.csv python -m ai_ml.data.load
python -m ai_ml.models.train
