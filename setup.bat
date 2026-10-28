@echo off
cd /d "%~dp0"
if not exist .venv (
    py -3.12 -m venv .venv || py -3 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
if not exist .env copy .env.example .env
echo.
echo Setup finished. Next: train.bat
