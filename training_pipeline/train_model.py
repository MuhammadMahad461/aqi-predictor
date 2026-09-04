import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))
from hopsworks_utils import get_feature_store

def load_training_data():
    fs = get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=1)
    df = aqi_fg.read(read_options={"use_hive": True})
    return df

from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week",
    "temp", "humidity", "pressure", "wind_speed",
    "pm2_5", "pm10", "no2", "o3", "co", "so2", "nh3",
    "aqi_us", "aqi_change_rate",
]
TARGET_COL = "target_aqi_3d"

def prepare_data(df):
    df = df.sort_values("timestamp").dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n{name} — RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
    return {"model": model, "name": name, "rmse": rmse, "mae": mae, "r2": r2}

def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = []

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate_model(ridge, X_test, y_test, "Ridge Regression"))

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))

    return results

def select_best_model(results):
    best = min(results, key=lambda r: r["rmse"])
    print(f"\nBest model: {best['name']} (RMSE: {best['rmse']:.2f})")
    return best

def register_model(project, best_result, X_train):
    mr = project.get_model_registry()

    os.makedirs("training_pipeline/model_dir", exist_ok=True)
    model_path = "training_pipeline/model_dir/model.pkl"
    joblib.dump(best_result["model"], model_path)

    input_example = X_train.iloc[[0]]

    aqi_model = mr.python.create_model(
        name="aqi_forecast_model",
        metrics={"rmse": best_result["rmse"], "mae": best_result["mae"], "r2": best_result["r2"]},
        description=f"AQI 3-day forecast model ({best_result['name']})",
        input_example=input_example,
    )
    aqi_model.save("training_pipeline/model_dir")
    print(f"Model registered: {aqi_model.name}, version {aqi_model.version}")

if __name__ == "__main__":
    import hopsworks
    from dotenv import load_dotenv
    load_dotenv()

    print("Logging in...", flush=True)
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )
    print("Logged in.", flush=True)

    df = load_training_data()
    print(f"Loaded {len(df)} rows from feature store.", flush=True)
    print(df.head(), flush=True)

    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows", flush=True)

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: not enough data after dropna — check target_aqi_3d column exists and has values.")
        sys.exit(1)

    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    best = select_best_model(results)
    print("Registering model...", flush=True)
    register_model(project, best, X_train)
    print("Done.", flush=True)
