"""
Backend API for the sustainability advisor prototype.
Uses SQLite instead of the originally planned PostgreSQL - easier to set up
for a solo project.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import xgboost as xgb
import shap
import numpy as np
import json
import os
from datetime import datetime, timezone
import uuid

from train_model import calculate_carbon_footprint, RECOMMENDATIONS, DEFRA_FACTORS

app = Flask(__name__)
CORS(app)

DB_PATH = os.path.join(os.path.dirname(__file__), "sustainability.db")
FEATURE_NAMES = ["weekly_car_km", "weekly_electricity_kwh", "weekly_meat_meals",
                  "household_size", "is_rural"]

# rough sanity limits so a typo doesn't break everything
INPUT_BOUNDS = {
    "weekly_car_km": (0, 2000),
    "weekly_electricity_kwh": (0, 1000),
    "weekly_meat_meals": (0, 50),
    "household_size": (1, 20),
}

model = xgb.Booster()
model.load_model(os.path.join(os.path.dirname(__file__), "model.json"))
explainer = shap.TreeExplainer(model)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Keeps personal identifiers separate from behavioural data, linked only
    by a pseudonymous user_id, so the behaviour logs alone don't identify anyone.
    """
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        region TEXT DEFAULT 'uk',
        household_size INTEGER DEFAULT 1,
        is_rural INTEGER DEFAULT 0,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS behaviour_logs (
        log_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        weekly_car_km REAL,
        weekly_electricity_kwh REAL,
        weekly_meat_meals REAL,
        footprint_kgco2e REAL,
        logged_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    CREATE TABLE IF NOT EXISTS recommendation_events (
        event_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        recommendation_id TEXT,
        recommendation_text TEXT,
        accepted INTEGER,
        shap_explanation TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)
    conn.commit()
    conn.close()


def validate_and_parse(data: dict):
    """
    Validate incoming request data. Returns (parsed_dict, error_message).
    error_message is None if validation passed. Keeps the demo from crashing
    ugly on a mistyped or missing field.
    """
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    parsed = {}
    for field in ["weekly_car_km", "weekly_electricity_kwh", "weekly_meat_meals"]:
        raw = data.get(field, 0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"'{field}' must be a number, got: {raw!r}"
        lo, hi = INPUT_BOUNDS[field]
        if not (lo <= value <= hi):
            return None, f"'{field}' must be between {lo} and {hi}, got: {value}"
        parsed[field] = value

    raw_household = data.get("household_size", 1)
    try:
        household_size = int(raw_household)
    except (TypeError, ValueError):
        return None, f"'household_size' must be a whole number, got: {raw_household!r}"
    lo, hi = INPUT_BOUNDS["household_size"]
    if not (lo <= household_size <= hi):
        return None, f"'household_size' must be between {lo} and {hi}, got: {household_size}"
    parsed["household_size"] = household_size

    is_rural = data.get("is_rural", 0)
    parsed["is_rural"] = 1 if str(is_rural) in ("1", "true", "True") else 0

    region = data.get("region", "uk")
    if region not in ("uk", "scotland"):
        return None, f"'region' must be 'uk' or 'scotland', got: {region!r}"
    parsed["region"] = region

    parsed["user_id"] = data.get("user_id") or str(uuid.uuid4())
    return parsed, None


def get_recommendations(user_data: dict, shap_values: np.ndarray):
    """Select applicable recommendations and rank by SHAP-driven feature importance."""
    feature_importance = dict(zip(FEATURE_NAMES, shap_values.tolist()))
    applicable = []
    for rec in RECOMMENDATIONS:
        user_value = user_data.get(rec["trigger_feature"], 0)
        if user_value >= rec["threshold"]:
            importance = feature_importance.get(rec["trigger_feature"], 0)
            applicable.append({**rec, "importance": importance})
    applicable.sort(key=lambda r: r["importance"], reverse=True)
    return applicable[:3]


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found. Check the endpoint path."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error. Check the server console for details."}), 500


@app.route("/api/advice", methods=["POST"])
def get_advice():
    """
    Main endpoint. Takes weekly behaviour data, works out the footprint,
    runs it through the model, gets SHAP explanations, and returns ranked
    recommendations with plain-English reasons attached.
    """
    try:
        raw_data = request.get_json(force=True, silent=True)
    except Exception:
        raw_data = None
    if raw_data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    parsed, error = validate_and_parse(raw_data)
    if error:
        return jsonify({"error": error}), 400

    user_data = {
        "weekly_car_km": parsed["weekly_car_km"],
        "weekly_electricity_kwh": parsed["weekly_electricity_kwh"],
        "weekly_meat_meals": parsed["weekly_meat_meals"],
    }

    try:
        footprint = calculate_carbon_footprint(user_data, region=parsed["region"])

        features = np.array([[
            parsed["weekly_car_km"], parsed["weekly_electricity_kwh"],
            parsed["weekly_meat_meals"], parsed["household_size"], parsed["is_rural"]
        ]])
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
        acceptance_prob = float(model.predict(dmatrix)[0])

        shap_values = explainer.shap_values(features)[0]
        recommendations = get_recommendations(
            {**user_data, "household_size": parsed["household_size"], "is_rural": parsed["is_rural"]},
            shap_values
        )
    except Exception as exc:
        app.logger.exception("Model inference failed")
        return jsonify({"error": f"Model inference failed: {exc}"}), 500

    try:
        conn = get_db()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id, region, household_size, is_rural, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (parsed["user_id"], parsed["region"], parsed["household_size"], parsed["is_rural"], now)
        )
        conn.execute(
            "INSERT INTO behaviour_logs (log_id, user_id, weekly_car_km, weekly_electricity_kwh, "
            "weekly_meat_meals, footprint_kgco2e, logged_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), parsed["user_id"], parsed["weekly_car_km"],
             parsed["weekly_electricity_kwh"], parsed["weekly_meat_meals"], footprint, now)
        )
        for rec in recommendations:
            conn.execute(
                "INSERT INTO recommendation_events (event_id, user_id, recommendation_id, "
                "recommendation_text, accepted, shap_explanation, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), parsed["user_id"], rec["id"], rec["text"], None,
                 json.dumps({"importance": rec["importance"]}), now)
            )
        conn.commit()
        conn.close()
    except sqlite3.Error as exc:
        app.logger.exception("Database write failed")
        return jsonify({"error": f"Database write failed: {exc}"}), 500

    return jsonify({
        "user_id": parsed["user_id"],
        "weekly_footprint_kgco2e": footprint,
        "acceptance_likelihood": round(acceptance_prob, 3),
        "recommendations": [
            {
                "id": r["id"],
                "category": r["category"],
                "text": r["text"],
                "explanation": f"Suggested because your {r['trigger_feature'].replace('_', ' ')} "
                                f"was the strongest contributing factor to your footprint"
            } for r in recommendations
        ]
    })


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Record whether a user accepted a recommendation (for future model retraining)."""
    data = request.get_json(force=True, silent=True) or {}
    event_id = data.get("event_id")
    if not event_id:
        return jsonify({"error": "'event_id' is required."}), 400
    try:
        conn = get_db()
        conn.execute(
            "UPDATE recommendation_events SET accepted = ? WHERE event_id = ?",
            (int(bool(data.get("accepted", 0))), event_id)
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as exc:
        return jsonify({"error": f"Database update failed: {exc}"}), 500
    return jsonify({"status": "recorded"})


@app.route("/api/history/<user_id>", methods=["GET"])
def history(user_id):
    """Lets a user pull their own logged data back out, GDPR-style."""
    try:
        conn = get_db()
        logs = conn.execute(
            "SELECT * FROM behaviour_logs WHERE user_id = ? ORDER BY logged_at DESC", (user_id,)
        ).fetchall()
        conn.close()
    except sqlite3.Error as exc:
        return jsonify({"error": f"Database read failed: {exc}"}), 500
    return jsonify([dict(row) for row in logs])


@app.route("/api/reset", methods=["POST"])
def reset_demo_data():
    """Wipes the logged data so I can demo this more than once without the history piling up."""
    conn = get_db()
    conn.execute("DELETE FROM recommendation_events")
    conn.execute("DELETE FROM behaviour_logs")
    conn.execute("DELETE FROM users")
    conn.commit()
    conn.close()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    init_db()
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5001, debug=debug_mode)
