@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
mlflow ui --backend-store-uri sqlite:///mlflow.db
