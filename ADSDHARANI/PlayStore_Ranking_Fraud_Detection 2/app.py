from pathlib import Path
import math
from urllib.parse import urlparse, parse_qs
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
try:
    from google_play_scraper import app as fetch_play_store_app
except ImportError:
    fetch_play_store_app = None

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder="templates", static_folder="static")
MODEL_PATH = ROOT / "models" / "playstore_rating_model.pkl"
DATA_PATH = ROOT / "data" / "googleplaystore_cleaned.csv"

def get_bundle():
    if not MODEL_PATH.exists():
        return None
    # Only load the project model bundled with this trusted application.
    return joblib.load(MODEL_PATH)

def get_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

def score_pattern(rating, reviews, installs):
    """Transparent ranking-manipulation screening rules; not a malware/safety scan."""
    flags = []
    if rating >= 4.8 and reviews < 20:
        flags.append("Very high rating with very few reviews")
    if installs > 0 and reviews > installs * 0.25 and reviews >= 100:
        flags.append("Review count is unusually high relative to installs")
    if reviews == 0 and rating >= 4.5:
        flags.append("High rating with no recorded reviews")
    if installs >= 10000 and reviews < 5:
        flags.append("Many installs but very few reviews")
    if flags:
        return "Potentially suspicious — manual review recommended", flags
    return "No obvious ranking-fraud signals found", ["No configured heuristic was triggered. This does not prove the app is safe or fraud-free."]

def extract_play_store_id(url):
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() not in ("play.google.com", "www.play.google.com"):
        raise ValueError("Please paste a valid Google Play Store app URL (play.google.com/store/apps/details?id=...).")
    if not parsed.path.startswith("/store/apps/details"):
        raise ValueError("This link does not look like a Google Play app details page.")
    app_id = parse_qs(parsed.query).get("id", [""])[0].strip()
    if not app_id or len(app_id) > 250:
        raise ValueError("Could not find the app ID in that link. Copy the app's full Google Play URL.")
    return app_id

def analyze_play_store_url(url):
    if fetch_play_store_app is None:
        raise ValueError("Google Play lookup library is not installed. Run: python -m pip install -r requirements.txt")
    app_id = extract_play_store_id(url)
    try:
        info = fetch_play_store_app(app_id, lang="en", country="us")
    except Exception as exc:
        raise ValueError("Could not retrieve this app from Google Play. Check the URL and internet connection, then try again.") from exc
    rating = float(info.get("score") or 0)
    reviews = int(info.get("reviews") or 0)
    installs_text = str(info.get("minInstalls") or 0)
    installs = int(info.get("minInstalls") or 0)
    size_bytes = info.get("size")
    size_mb = round(float(size_bytes) / (1024 * 1024), 2) if isinstance(size_bytes, (int, float)) and size_bytes > 0 else 0.0
    price = float(info.get("price") or 0)
    status, flags = score_pattern(rating, reviews, installs)
    bundle = get_bundle()
    predicted = None
    if bundle is not None:
        X = pd.DataFrame([{"Log_Reviews": np.log1p(reviews), "Log_Installs": np.log1p(installs), "Price": price, "Size_MB": size_mb}], columns=bundle["features"])
        predicted = round(float(np.clip(bundle["model"].predict(X)[0], 0, 5)), 2)
    return {
        "app_name": info.get("title") or app_id,
        "app_id": app_id,
        "developer": info.get("developer") or "Not listed",
        "actual_rating": round(rating, 2),
        "reviews": reviews,
        "installs": installs_text,
        "price": price,
        "size_mb": size_mb,
        "predicted_rating": predicted,
        "status": status,
        "flags": flags,
        "url": url.strip(),
    }

@app.route("/")
def home():
    df = get_data()
    return render_template("index.html", active="home", total_apps=len(df),
                           categories=df["Category"].nunique() if "Category" in df else 0)

@app.route("/dashboard")
def dashboard():
    df = get_data()
    if df.empty:
        return render_template("dashboard.html", active="dashboard", stats={}, categories=[], top_apps=[])
    for col in ["Rating", "Reviews", "Installs", "Price"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    stats = {
        "apps": int(len(df)),
        "avg_rating": round(float(df["Rating"].mean()), 2) if "Rating" in df else 0,
        "total_reviews": int(df["Reviews"].fillna(0).sum()) if "Reviews" in df else 0,
        "categories": int(df["Category"].nunique()) if "Category" in df else 0
    }
    cats = (df["Category"].value_counts().head(7).to_dict() if "Category" in df else {})
    top = []
    if "App" in df and "Rating" in df:
        top = df.sort_values("Rating", ascending=False)[["App","Rating"]].head(6).to_dict("records")
    return render_template("dashboard.html", active="dashboard", stats=stats,
                           categories=list(cats.items()), top_apps=top)

@app.route("/analyzer", methods=["GET", "POST"])
def analyzer():
    result = None
    error = None
    if request.method == "POST":
        try:
            url = request.form.get("play_store_url", "").strip()
            if not url:
                raise ValueError("Paste a Google Play Store app link to analyze it.")
            result = analyze_play_store_url(url)
        except ValueError as exc:
            error = str(exc)
    return render_template("analyzer.html", active="analyzer", result=result, error=error)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True) or {}
    try:
        rating = float(payload.get("rating", 0))
        reviews = int(payload.get("reviews", 0))
        installs = int(payload.get("installs", 0))
        price = float(payload.get("price", 0))
        size_mb = float(payload.get("size_mb", 0))
        if not (0 <= rating <= 5 and min(reviews, installs, price, size_mb) >= 0):
            return jsonify({"error": "Values must be non-negative and rating must be between 0 and 5."}), 400
        bundle = get_bundle()
        if bundle is None:
            return jsonify({"error": "Model missing; run train_model.py."}), 500
        X = pd.DataFrame([{"Log_Reviews": np.log1p(reviews), "Log_Installs": np.log1p(installs),
                           "Price": price, "Size_MB": size_mb}], columns=bundle["features"])
        predicted = float(np.clip(bundle["model"].predict(X)[0], 0, 5))
        status, flags = score_pattern(rating, reviews, installs)
        return jsonify({"predicted_rating": round(predicted, 2), "status": status, "flags": flags,
                        "notice": "Heuristic flags are for investigation only and do not confirm fraud."})
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input values."}), 400

@app.route("/visualizations")
def visualizations():
    df = get_data()
    charts = {}
    if not df.empty:
        if "Rating" in df:
            charts["ratings"] = df["Rating"].dropna().round(1).value_counts().sort_index().to_dict()
        if "Category" in df:
            charts["categories"] = df["Category"].value_counts().head(8).to_dict()
    bundle = get_bundle()
    metrics = bundle.get("metrics", {}) if bundle else {}
    return render_template("visualizations.html", active="visualizations", charts=charts, metrics=metrics)

@app.route("/dataset")
def dataset():
    df = get_data()
    return render_template("dataset.html", active="dataset", columns=list(df.columns),
                           rows=df.head(12).fillna("").to_dict("records"), row_count=len(df))

@app.route("/download/cleaned")
def download_cleaned():
    return send_from_directory(DATA_PATH.parent, DATA_PATH.name, as_attachment=True)

@app.route("/download/raw")
def download_raw():
    return send_from_directory(ROOT / "data", "googleplaystore_raw.csv", as_attachment=True)

@app.route("/about")
def about():
    return render_template("about.html", active="about")

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
