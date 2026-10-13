"""Train a demo rating-regression model and save a trusted local pickle bundle."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "googleplaystore_cleaned.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
required = ["Rating", "Reviews", "Installs", "Price"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Required column missing from dataset: {col}")

for col in ["Rating", "Reviews", "Installs", "Price"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if "Size_MB" not in df.columns:
    def size_mb(value):
        try:
            value = str(value).strip()
            if value.endswith("M"): return float(value[:-1])
            if value.endswith("k"): return float(value[:-1]) / 1024
        except (ValueError, TypeError):
            pass
        return np.nan
    df["Size_MB"] = df["Size"].apply(size_mb) if "Size" in df else np.nan
df["Size_MB"] = pd.to_numeric(df["Size_MB"], errors="coerce")
df = df[df["Rating"].between(0, 5)].copy()
df["Reviews"] = df["Reviews"].clip(lower=0)
df["Installs"] = df["Installs"].clip(lower=0)
df["Price"] = df["Price"].clip(lower=0)
df["Size_MB"] = df["Size_MB"].clip(lower=0)
df = df.dropna(subset=["Rating"])
if len(df) < 5:
    raise ValueError("At least 5 rows with valid ratings are required to train this demo model.")

df["Log_Reviews"] = np.log1p(df["Reviews"].fillna(df["Reviews"].median()))
df["Log_Installs"] = np.log1p(df["Installs"].fillna(df["Installs"].median()))
features = ["Log_Reviews", "Log_Installs", "Price", "Size_MB"]
X = df[features]
y = df["Rating"]

# Small demo datasets use a split only when there are enough rows.
test_size = 0.25 if len(df) >= 12 else 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regression", LinearRegression())
])
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
metrics = {
    "r2": float(r2_score(y_test, pred)) if len(y_test) > 1 else None,
    "mse": float(mean_squared_error(y_test, pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "rows_used": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "features": features,
    "target": "Rating",
    "note": "Demo metrics from a very small synthetic practice dataset; not evidence of real-world model performance."
}
bundle = {
    "model": pipeline,
    "features": features,
    "metrics": metrics,
    "training_rows": int(len(df)),
    "target": "Rating",
    "dataset_type": "synthetic practice data"
}
joblib.dump(bundle, MODEL_DIR / "playstore_rating_model.pkl")
(MODEL_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
print("Model saved to models/playstore_rating_model.pkl")
print(json.dumps(metrics, indent=2))
