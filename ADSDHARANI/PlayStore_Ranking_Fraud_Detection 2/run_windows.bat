@echo off
cd /d "%~dp0"
if not exist .venv (
  py -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python app.py
pause
