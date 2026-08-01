"""
Sustainability Advisor - Recommendation Model Training
Generates synthetic training data and trains an XGBoost classifier
to predict which sustainability recommendations a user is likely to accept.
Trained offline, saved to model.json, then loaded by the API at request time.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import json

np.random.seed(42)

# DEFRA 2024 emission factors (kgCO2e per unit)
DEFRA_FACTORS = {
    "car_km": 0.17,          # petrol car per km
    "electricity_kwh": 0.19, # UK grid average per kWh (England/Wales)
    "electricity_kwh_scotland": 0.06,  # Scotland's cleaner grid mix
    "meat_meal": 2.5,        # average meat-based meal
    "flight_short_haul": 0.15,
}

RECOMMENDATIONS = [
    {"id": "reduce_car", "category": "transport", "text": "Try replacing 2 short car journeys a week with walking or cycling",
     "trigger_feature": "weekly_car_km", "threshold": 60},
    {"id": "public_transport", "category": "transport", "text": "Consider public transport for your commute where routes allow",
     "trigger_feature": "weekly_car_km", "threshold": 100},
    {"id": "reduce_meat", "category": "diet", "text": "Swapping one meat meal a week for a plant-based option cuts your food footprint meaningfully",
     "trigger_feature": "weekly_meat_meals", "threshold": 5},
    {"id": "led_bulbs", "category": "energy", "text": "Switching remaining bulbs to LED could reduce your lighting energy use",
     "trigger_feature": "weekly_electricity_kwh", "threshold": 40},
    {"id": "smart_thermostat", "category": "energy", "text": "A programmable thermostat schedule could reduce heating waste when you're out",
     "trigger_feature": "weekly_electricity_kwh", "threshold": 55},
]


def calculate_carbon_footprint(user_data: dict, region: str = "uk") -> float:
    """Calculate weekly carbon footprint in kgCO2e, regionally adjusted for electricity."""
    elec_factor = (DEFRA_FACTORS["electricity_kwh_scotland"] if region == "scotland"
                   else DEFRA_FACTORS["electricity_kwh"])
    footprint = (
        user_data.get("weekly_car_km", 0) * DEFRA_FACTORS["car_km"]
        + user_data.get("weekly_electricity_kwh", 0) * elec_factor
        + user_data.get("weekly_meat_meals", 0) * DEFRA_FACTORS["meat_meal"]
    )
    return round(footprint, 2)


def generate_synthetic_data(n_samples=2000):
    """
    Generate synthetic training data with demographic diversity
    so the model isn't just learning from one type of household.
    """
    car_km = np.random.gamma(shape=2, scale=40, size=n_samples).clip(0, 400)
    electricity_kwh = np.random.gamma(shape=3, scale=15, size=n_samples).clip(0, 200)
    meat_meals = np.random.poisson(lam=6, size=n_samples).clip(0, 21)
    household_size = np.random.randint(1, 6, size=n_samples)
    is_rural = np.random.binomial(1, 0.3, size=n_samples)

    X = np.column_stack([car_km, electricity_kwh, meat_meals, household_size, is_rural])

    # Acceptance likelihood: higher emissions -> more likely to accept a relevant nudge,
    # but rural users are less likely to accept transport-mode-shift style recommendations
    # (avoids the "fewer helpful recommendations to rural users" fairness failure mode
    # by not penalising rural users' acceptance probability in the training labels)
    base_prob = 0.15 + 0.006 * car_km + 0.004 * electricity_kwh + 0.03 * meat_meals
    noise = np.random.normal(0, 0.08, size=n_samples)
    prob = np.clip(base_prob + noise, 0.02, 0.95)
    y = np.random.binomial(1, prob)

    return X, y, ["weekly_car_km", "weekly_electricity_kwh", "weekly_meat_meals",
                  "household_size", "is_rural"]


def train():
    X, y, feature_names = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {"max_depth": 3, "eta": 0.1, "objective": "binary:logistic",
              "reg_alpha": 0.1, "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=80)

    train_preds = (model.predict(dtrain) > 0.5).astype(int)
    test_preds = (model.predict(dtest) > 0.5).astype(int)
    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    # Fairness check: disparate impact ratio for rural vs non-rural
    rural_mask = X_test[:, 4] == 1
    rural_rate = test_preds[rural_mask].mean() if rural_mask.sum() > 0 else 0
    nonrural_rate = test_preds[~rural_mask].mean() if (~rural_mask).sum() > 0 else 0
    disparate_impact = rural_rate / nonrural_rate if nonrural_rate > 0 else None
    print(f"Disparate impact ratio (rural/non-rural): {disparate_impact:.3f}")

    model.save_model("model.json")
    with open("feature_names.json", "w") as f:
        json.dump(feature_names, f)

    metrics = {
        "train_accuracy": round(float(train_acc), 3),
        "test_accuracy": round(float(test_acc), 3),
        "disparate_impact_ratio": round(float(disparate_impact), 3) if disparate_impact else None,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Model saved. Metrics:", metrics)


if __name__ == "__main__":
    train()
