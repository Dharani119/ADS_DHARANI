# PlayGuard — Ranking Fraud Detection in Google Play Store Apps

A Flask-based educational data science project using a white-and-blue interface, a saved scikit-learn model (`.pkl`), a dashboard, dataset preview/download, and an interactive Google Play URL analyzer that retrieves public listing metrics and screens for unusual ranking/review patterns.

## Included
- `app.py` — Flask web application
- `train_model.py` — train and save the rating prediction model
- `models/playstore_rating_model.pkl` — trained model bundle (created by training script)
- `data/googleplaystore_raw.csv` — supplied practice dataset
- `data/googleplaystore_cleaned.csv` — supplied cleaned dataset
- `notebooks/FraudDetectionDataProcessed.ipynb` — preprocessing and regression notebook supplied for this project
- `templates/`, `static/` — white-and-blue responsive web interface
- `run_windows.bat` — easy Windows setup and launch

## Run on Windows
1. Extract the ZIP file.
2. Install Python 3.10 or newer.
3. Double-click `run_windows.bat`, or open Command Prompt in this folder and run:
   ```bat
   py -m venv .venv
   .venv\\Scripts\\activate
   python -m pip install -r requirements.txt
   python train_model.py
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

If `py` is not recognized, try `python` instead. Internet is required once to install dependencies and Chart.js/Google Fonts are loaded from CDN when online.

## Model
The saved scikit-learn pipeline uses Multiple Linear Regression to predict an app's rating from log-transformed review count, log-transformed installs, price, and app size. It reports R², MSE, and RMSE on a held-out split.

## Important limitations
The provided practice data is **synthetic and very small** (20 raw rows; 18 rows in the supplied cleaned CSV). It has no verified fraud labels, no historical rank series, and no review timestamps. The web interface therefore presents transparent heuristic patterns for human review; it does not claim to prove an app is fraudulent. Regression predicts rating, not fraud. Replace the practice data with a sourced, sufficiently large real dataset and validate detection against reliable labels before making research claims.

Only load pickle/joblib files you trust, because model serialization formats can execute code when loaded.
